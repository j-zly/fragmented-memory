"""keepsake v2 — 写侧两相管线（Mem0 风格 + 封边取代）。

# 为什么做（2026-09-08 事故复盘的根因层）

v1 写闸门（ingest_gate）是正则垃圾过滤器：黑名单追不上语言变体，
一刀切长度会误杀长事实，**且存的是消息原文**——需求原话被存成永恒事实，
后续无人回改，导致「7 月已完成的分页需求被当待办复活」事故。

业界做法（Mem0 arXiv:2504.19413）= LLM 两相：

  * 提取相：从对话提炼「脱离本对话仍有长期价值」的 salient facts。
    废话自然无事实可提而消亡，不需黑名单。
  * 更新相：拿新事实对已有记忆做 ADD/UPDATE/DELETE/NOOP。

Zep/Graphiti 进一步：矛盾时旧记忆**不物理删而是封边**——记
superseded_by/superseded_at 时间窗，检索默认只出当前有效。

# 设计

  Pipeline  = 异步 daemon + 双相 LLM
  ├─ 队列    deque[(user, assistant, ts)]
  ├─ 线程    daemon=True；窗口触发：≥window_pairs 对 OR 距上次 drain ≥ window_seconds
  ├─ 提取相  一次 LLM（输入=本窗口 + 最近 8 条环形缓冲文本）→ JSON facts
  └─ 更新相  每条 fact 用 search_bm25 取 top-5（已自然排除 consumed/superseded）→
              一次 LLM 四选一（ADD / UPDATE / DELETE / NOOP）
              UPDATE：新碎片 supersedes=<old>；旧碎片只打 superseded_by 标记
              DELETE：旧碎片 superseded_by='__void__'

# 故障兜底

  * LLM 提取/更新任一调用失败或 JSON 解析失败 → 窗口整体回落 v1 decide() 路径
  * 窗口内 LLM 调用数 > max_calls_per_window → 回落
  * 兜底保证：**绝不静默丢消息**

# 隔离

  * 不 import redis —— 存储操作全走注入的 storage 对象（带 mock 测试便利）
  * _call_llm 通过参数注入 —— 测试可 monkeypatch，不连真实 LLM
  * 启动/排空由 KeepsakeProvider 显式调用，不与 Hermes runtime 隐式耦合
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "model": "qwen-plus",         # 默认复用 consolidator 的 DEFAULT_LLM_MODEL
    "window_pairs": 4,
    "window_seconds": 30.0,
    "max_calls_per_window": 8,
    "update_top_k": 5,            # UPDATE 阶段每条 fact 取的相似旧碎片数
    "recent_context_size": 8,     # 提取相的会话内环形缓冲大小
}

VALID_KINDS = frozenset({"fact", "decision", "preference", "pending", "state_change"})

VOID_KEY = "__void__"  # DELETE 路径专用 sentinel


# ---------------------------------------------------------------------------
# 提示词（few-shot 内嵌，注释里写出设计点）
# ---------------------------------------------------------------------------

# 提取相：要点写在注释里
#   1. 只提「脱离本对话仍有长期价值」的内容 —— 纯确认/寒暄/状态询问 → 空数组
#   2. kind ∈ {fact, decision, preference, pending, state_change}
#   3. few-shot 必须给出三类典型：
#      - "可以的" → 空数组（废话）
#      - "资金总览要做分页" → pending（任务/待办）
#      - "我选了A方案今晚部署" → decision（用户决定）
EXTRACT_SYSTEM = "你是一位知识提取专家，擅长从对话中提炼长期有价值的事实条目。"

EXTRACT_USER_TEMPLATE = """从以下对话窗口中提取"脱离本对话仍有长期价值"的事实条目。

规则：
1. 只提取「脱离本对话仍有长期价值」的内容
2. 纯确认（"可以的"）、寒暄、状态询问 → 返回空数组
3. kind 取值：fact / decision / preference / pending / state_change
4. 每条 fact 包含 content 和 kind
5. 输出严格 JSON，格式：`{{"facts":[{{"content":"...","kind":"..."}}]}}`

会话内最近 {recent_count} 条上下文（仅供语境参考，不要从中提事实）：
{recent_context}

本窗口对话（{window_count} 条 user/assistant 对）：
{window_turns}

few-shot：
- "可以的" → {{"facts":[]}}
- "资金总览要做分页" → {{"facts":[{{"content":"资金总览需要做分页","kind":"pending"}}]}}
- "我选了A方案今晚部署" → {{"facts":[{{"content":"用户决定选择A方案今晚部署","kind":"decision"}}]}}
"""


# 更新相：要点写在注释里
#   1. ADD：无等价物 → 写新碎片
#   2. UPDATE：同主题且有新信息 → 写新碎片，旧碎片只打 superseded_by 标记
#   3. DELETE：与旧碎片矛盾且新事实不一定占优 → 封旧（superseded_by=__void__）
#   4. NOOP：忽略
#   5. few-shot 必须含状态变更例：旧"用户要求做前端分页" + 新"分页已完成上线" → UPDATE 封旧
UPDATE_SYSTEM = "你是一位记忆整理专家，决定新事实与已有记忆的关系（ADD/UPDATE/DELETE/NOOP）。"

UPDATE_USER_TEMPLATE = """新事实：{new_fact}
kind: {new_kind}

已有相似记忆（top {top_k}）：
{candidates_block}

四选一：
- ADD：新事实无等价物 → 写新记忆
- UPDATE：新事实是同一主题的更新 → 写新记忆，旧记忆封边（superseded_by）
- DELETE：新事实与旧记忆矛盾但新事实不一定占优 → 封旧记忆（不留新事实）
- NOOP：新事实已被旧记忆覆盖 → 忽略

输出严格 JSON：`{{"action":"ADD|UPDATE|DELETE|NOOP","target_key":"<旧 key 或空>"}}`

few-shot：
- 旧"用户要求做前端分页" + 新"分页已完成上线并验证" → UPDATE 封旧
"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """单轮对话（user + assistant）。"""
    user: str
    assistant: str
    timestamp: float


@dataclass
class Fact:
    """提取相输出：单条事实。"""
    content: str
    kind: str  # fact / decision / preference / pending / state_change


@dataclass
class DrainResult:
    """单次 drain 的统计。"""
    window_turns: int = 0
    facts_extracted: int = 0
    adds: int = 0
    updates: int = 0
    deletes: int = 0
    noops: int = 0
    llm_calls: int = 0
    fallback_v1: bool = False
    fallback_reason: str = ""


# ---------------------------------------------------------------------------
# 核心：Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """写侧两相管线。

    调用方（KeepsakeProvider）负责：
      * initialize() 时 start()
      * 每轮对话 enqueue(user, assistant)
      * shutdown() 时 stop(drain=True)

    Pipeline 自身不连 Redis（依赖注入的 storage），不调真实 LLM
    （_call_llm 通过构造函数注入，单测可 monkeypatch）。
    """

    def __init__(
        self,
        storage: Any,
        *,
        llm_fn: Optional[Callable[..., Optional[str]]] = None,
        model: str = DEFAULT_PIPELINE_CONFIG["model"],
        window_pairs: int = DEFAULT_PIPELINE_CONFIG["window_pairs"],
        window_seconds: float = DEFAULT_PIPELINE_CONFIG["window_seconds"],
        max_calls_per_window: int = DEFAULT_PIPELINE_CONFIG["max_calls_per_window"],
        update_top_k: int = DEFAULT_PIPELINE_CONFIG["update_top_k"],
        recent_context_size: int = DEFAULT_PIPELINE_CONFIG["recent_context_size"],
        gate_fallback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        参数:
            storage: RedisStorage 实例（必须实现 search_bm25/store/get_fragment/supersede_fragment/_get_client）
            llm_fn: LLM 调用函数，签名 (messages, model) -> Optional[str]；None 则全部走 v1 兜底
            model: LLM 模型名
            window_pairs: 队列攒够多少对触发 drain
            window_seconds: 距上次 drain 超过多少秒触发 drain
            max_calls_per_window: 单窗口 LLM 调用硬顶
            update_top_k: 更新相每条 fact 取的相似旧碎片数
            recent_context_size: 提取相注入的会话内环形缓冲大小
            gate_fallback: v1 兜底函数 (text, category) -> None（通常就是 sync_turn 走 decide() 的那段）
        """
        self._storage = storage
        self._llm_fn = llm_fn
        self._model = model
        self._window_pairs = int(window_pairs)
        self._window_seconds = float(window_seconds)
        self._max_calls = int(max_calls_per_window)
        self._update_top_k = int(update_top_k)
        self._recent_size = int(recent_context_size)
        self._gate_fallback = gate_fallback

        self._queue: Deque[Turn] = deque()
        self._lock = threading.Lock()
        self._recent_ring: Deque[Tuple[str, str]] = deque(maxlen=self._recent_size * 2)
        self._last_drain_ts: float = time.time()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动 daemon 线程。幂等。"""
        if self._started:
            return
        self._stop_event.clear()
        # 初始化 last_drain_ts 为当前时间，否则第一次 enqueue 会被 window_seconds 条件
        # 误判为「距上次 drain 超 30s」而立即 drain（首次启动不该这样）
        self._last_drain_ts = time.time()
        self._thread = threading.Thread(
            target=self._run, name="keepsake-pipeline", daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info("keepsake pipeline started (window_pairs=%d, window_seconds=%.0f)",
                    self._window_pairs, self._window_seconds)

    def stop(self, *, drain: bool = True, timeout: float = 5.0) -> None:
        """停止 daemon；可选先排空队列（shutdown 路径）。"""
        if drain:
            try:
                self._drain_now()
            except Exception as e:
                logger.warning("keepsake pipeline drain on shutdown failed: %s", e)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._started = False
        logger.info("keepsake pipeline stopped")

    def enqueue(self, user_text: str, assistant_text: str = "") -> None:
        """主路径调用（sync_turn 内）：入队 + 视情况立即 drain。"""
        if not user_text or not user_text.strip():
            return
        with self._lock:
            self._queue.append(Turn(
                user=user_text,
                assistant=assistant_text or "",
                timestamp=time.time(),
            ))
            self._recent_ring.append((user_text, assistant_text or ""))
            # 触发条件 1：攒够 window_pairs
            if len(self._queue) >= self._window_pairs:
                should_drain = True
            else:
                # 触发条件 2：距上次 drain 超 window_seconds
                should_drain = (time.time() - self._last_drain_ts) >= self._window_seconds
        if should_drain:
            self._drain_now()

    # ------------------------------------------------------------------
    # Daemon 主循环
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """daemon 主循环：每秒检查一次窗口条件。"""
        while not self._stop_event.is_set():
            try:
                # 检查条件（持锁拷贝必要状态）
                should_drain = False
                with self._lock:
                    if self._queue:
                        if len(self._queue) >= self._window_pairs:
                            should_drain = True
                        elif (time.time() - self._last_drain_ts) >= self._window_seconds:
                            should_drain = True
                if should_drain:
                    self._drain_now()
            except Exception as e:
                logger.warning("keepsake pipeline loop error: %s", e)
            # 节流 1s（细粒度足够；shutdown 信号最长 1s 延迟）
            self._stop_event.wait(timeout=1.0)

    # ------------------------------------------------------------------
    # 单次 drain
    # ------------------------------------------------------------------

    def _drain_now(self) -> DrainResult:
        """立即排空一次窗口。无 turn 也返回空结果。"""
        with self._lock:
            turns = list(self._queue)
            self._queue.clear()
            recent = list(self._recent_ring)
            self._last_drain_ts = time.time()
        if not turns:
            return DrainResult()
        return self._process_window(turns, recent)

    def _process_window(self, turns: List[Turn], recent: List[Tuple[str, str]]) -> DrainResult:
        """处理一个窗口：提取相 → 更新相；任何一步失败整体兜底 v1。"""
        result = DrainResult(window_turns=len(turns))

        # 0. 兜底前置：LLM 不可用 → 全部走 v1 兜底
        if self._llm_fn is None:
            return self._fallback_to_v1(turns, result, reason="llm_unavailable")

        # 1. 提取相
        try:
            facts = self._extract_phase(turns, recent)
        except _LLMFailure as e:
            return self._fallback_to_v1(turns, result, reason=f"extract_failed:{e}")
        except _BudgetExceeded:
            return self._fallback_to_v1(turns, result, reason="budget_exceeded")

        result.facts_extracted = len(facts)
        result.llm_calls += 1  # 提取相算一次

        if not facts:
            return result

        # 2. 更新相（每条 fact 一次 LLM）
        for fact in facts:
            if result.llm_calls >= self._max_calls:
                return self._fallback_to_v1(turns, result, reason="budget_exceeded")
            try:
                action, target_key = self._update_phase(fact)
            except _LLMFailure as e:
                return self._fallback_to_v1(turns, result, reason=f"update_failed:{e}")
            except _BudgetExceeded:
                return self._fallback_to_v1(turns, result, reason="budget_exceeded")
            result.llm_calls += 1

            if action == "ADD":
                self._do_add(fact)
                result.adds += 1
            elif action == "UPDATE":
                self._do_update(fact, target_key)
                result.updates += 1
            elif action == "DELETE":
                self._do_delete(fact, target_key)
                result.deletes += 1
            else:  # NOOP 或未知 action
                result.noops += 1

        return result

    # ------------------------------------------------------------------
    # 提取相
    # ------------------------------------------------------------------

    def _extract_phase(self, turns: List[Turn], recent: List[Tuple[str, str]]) -> List[Fact]:
        """一次 LLM 调用，返回 facts 列表。失败抛 _LLMFailure。"""
        recent_lines: List[str] = []
        for u, a in recent:
            recent_lines.append(f"user: {u[:200]}")
            if a:
                recent_lines.append(f"assistant: {a[:200]}")
        recent_text = "\n".join(recent_lines[-self._recent_size * 2:]) or "(无)"

        window_lines: List[str] = []
        for t in turns:
            window_lines.append(f"user: {t.user[:500]}")
            if t.assistant:
                window_lines.append(f"assistant: {t.assistant[:500]}")
        window_text = "\n".join(window_lines)

        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(
                recent_count=self._recent_size,
                recent_context=recent_text,
                window_count=len(turns),
                window_turns=window_text,
            )},
        ]
        try:
            raw = self._llm_fn(messages, self._model)  # type: ignore[misc]
        except Exception as e:
            raise _LLMFailure(f"extract LLM raised: {e}") from e
        if raw is None:
            raise _LLMFailure("llm returned None")
        facts = self._parse_facts(raw)
        return facts

    @staticmethod
    def _parse_facts(raw: str) -> List[Fact]:
        """LLM 输出 → Fact 列表。失败抛 _LLMFailure。"""
        # 兼容 LLM 在 JSON 外层包 ```json ... ``` 或前缀废话
        text = raw.strip()
        # 找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise _LLMFailure("no JSON object")
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise _LLMFailure(f"json decode error: {e}") from e
        facts_raw = obj.get("facts")
        if not isinstance(facts_raw, list):
            return []  # 空数组合法
        out: List[Fact] = []
        for item in facts_raw:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            kind = item.get("kind") or "fact"
            if not content or not isinstance(content, str):
                continue
            kind = str(kind).strip().lower()
            if kind not in VALID_KINDS:
                kind = "fact"  # 未知 kind 兜底为 fact
            out.append(Fact(content=content.strip(), kind=kind))
        return out

    # ------------------------------------------------------------------
    # 更新相
    # ------------------------------------------------------------------

    def _update_phase(self, fact: Fact) -> Tuple[str, Optional[str]]:
        """一次 LLM 调用：拿新 fact + 已有相似 → (action, target_key)。

        target_key 仅在 UPDATE/DELETE 时有意义（要封边哪个旧 key）；
        ADD/NOOP 时为 None —— 由执行函数按需 search 兜底。

        target_key 必须过 _sane_target 双校验（白名单 + exists）才返回；幻觉 key
        一律打 warning + 退回 None，让 _do_update/_do_delete 走 _find_supersede_target。
        """
        # 取 top-K 相似旧碎片（search_bm25 已经自然排除 consumed/superseded）
        candidates: List[Dict[str, Any]] = []
        try:
            results = self._storage.search_bm25(fact.content, tag_filter="")
            candidates = results[:self._update_top_k]
        except Exception as e:
            logger.debug("pipeline: search_bm25 for fact failed: %s", e)

        # 构造候选块
        if candidates:
            lines = []
            for i, c in enumerate(candidates, 1):
                key = c.get("_key", "?")
                content = c.get("content", "")[:200]
                kind = c.get("category", "?")
                lines.append(f"[{i}] key={key} kind={kind}\n    {content}")
            candidates_block = "\n".join(lines)
        else:
            candidates_block = "(无相似旧记忆)"

        messages = [
            {"role": "system", "content": UPDATE_SYSTEM},
            {"role": "user", "content": UPDATE_USER_TEMPLATE.format(
                new_fact=fact.content[:300],
                new_kind=fact.kind,
                top_k=self._update_top_k,
                candidates_block=candidates_block,
            )},
        ]
        try:
            raw = self._llm_fn(messages, self._model)  # type: ignore[misc]
        except Exception as e:
            raise _LLMFailure(f"update LLM raised: {e}") from e
        if raw is None:
            raise _LLMFailure("llm returned None")
        action, raw_target_key = self._parse_action(raw)

        # 校验 LLM 给的 target_key：白名单 + exists 双保险；任一不过 → 拒为幻觉 key
        if raw_target_key is not None and action in ("UPDATE", "DELETE"):
            sane_key = self._sane_target(raw_target_key, candidates, raw_from_llm=True)
            if sane_key is None:
                logger.warning(
                    "pipeline: rejected hallucinated target_key fact=%r key=%s",
                    fact.content[:60], raw_target_key,
                )
                raw_target_key = None

        return action, raw_target_key

    @staticmethod
    def _parse_action(raw: str) -> Tuple[str, Optional[str]]:
        """LLM 输出 → (action, target_key)。

        严格策略：只接受合法 JSON。JSON 缺失/解析失败/action 不在
        {ADD, UPDATE, DELETE, NOOP} 内 → 一律返回 ("NOOP", None)。

        拒绝文本关键字兜底——「无需 UPDATE」这类解释性废话会误触发 UPDATE 封边。
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return ("NOOP", None)
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return ("NOOP", None)
        action = str(obj.get("action", "")).strip().upper()
        tk = obj.get("target_key")
        target_key: Optional[str] = None
        if isinstance(tk, str) and tk:
            target_key = tk
        if action not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            return ("NOOP", None)
        return action, target_key

    # ------------------------------------------------------------------
    # 写入 / 封边（不 import redis，全走注入 storage）
    # ------------------------------------------------------------------

    def _do_add(self, fact: Fact) -> None:
        """ADD：直接写新碎片（category=fact_v2 让消费侧可识别）。"""
        try:
            self._storage.store(
                text=fact.content,
                tags=self._tags_for(fact),
                category="fact_v2",
                source="pipeline_v2",
                fragment_type="memory",
            )
        except Exception as e:
            logger.warning("pipeline: ADD failed for fact %r: %s", fact.content[:60], e)

    def _do_update(self, fact: Fact, target_key: Optional[str] = None) -> None:
        """UPDATE：写新碎片 supersedes=<old>；旧碎片 superseded_by=<new>。

        target_key 由 _update_phase 的 LLM 提供（已过 _sane_target 校验）；
        若为 None（LLM 没给 / 幻觉被拒）→ 走 _find_supersede_target（同样过 exists 校验）。
        """
        old_key = target_key or self._find_supersede_target(fact.content)
        if not old_key:
            # 找不到候选 → 退化 ADD（避免静默丢）
            self._do_add(fact)
            return
        try:
            self._storage.store(
                text=fact.content,
                tags=self._tags_for(fact),
                category="fact_v2",
                source="pipeline_v2",
                fragment_type="memory",
            )
        except Exception as e:
            logger.warning("pipeline: UPDATE store failed: %s", e)
            return
        new_key = self._key_for_content(fact.content)
        if new_key:
            try:
                # 新碎片加 supersedes 字段（store 不支持 → 直接 hset）
                client = self._storage._get_client()  # noqa: SLF001
                if client:
                    client.hset(new_key, "supersedes", old_key)
            except Exception as e:
                logger.debug("pipeline: set supersedes field failed: %s", e)
            try:
                self._storage.supersede_fragment(old_key, new_key)
            except Exception as e:
                logger.warning("pipeline: supersede_fragment failed: %s", e)

    def _do_delete(self, fact: Fact, target_key: Optional[str] = None) -> None:
        """DELETE：封旧（superseded_by=__void__），不写新事实。"""
        old_key = target_key or self._find_supersede_target(fact.content)
        if not old_key:
            return
        try:
            self._storage.supersede_fragment(old_key, VOID_KEY)
        except Exception as e:
            logger.warning("pipeline: DELETE failed: %s", e)

    def _find_supersede_target(self, fact_content: str) -> Optional[str]:
        """取 search_bm25 top 结果作为封边目标。带 exists 校验（防 LLM 幻觉 key 盲封）。

        search_bm25 自然排除 consumed/superseded；若首个结果存在（客户端可用时
        额外确认 exists），返回其 _key，否则继续向后扫；都不可用 → None。
        """
        try:
            results = self._storage.search_bm25(fact_content, tag_filter="")
        except Exception as e:
            logger.debug("pipeline: _find_supersede_target failed: %s", e)
            return None
        for c in results or []:
            key = c.get("_key")
            if not key:
                continue
            if self._key_exists(key):
                return key
        return None

    def _sane_target(
        self,
        target_key: Optional[str],
        candidates: List[Dict[str, Any]],
        *,
        raw_from_llm: bool,
    ) -> Optional[str]:
        """校验 target_key 是否可用（白名单 + exists 双保险）。

        参数:
            target_key: 待校验的 key（LLM 给的或 _find_supersede_target 返回的）
            candidates: 本次 update_phase 给 LLM 看的候选列表（白名单源）
            raw_from_llm: True=LLM 直接给的；False=内部 _find 兜底结果（白名单可放宽）

        规则:
          - target_key 为空 → None
          - raw_from_llm=True 且 key 不在 candidates 白名单 → None
          - exists(target_key) 为假 → None（client 不可用时容错放行）
          - 都通过 → 原样返回
        """
        if not target_key:
            return None
        if raw_from_llm:
            allowed = {c.get("_key") for c in candidates if c.get("_key")}
            if allowed and target_key not in allowed:
                return None
        if not self._key_exists(target_key):
            return None
        return target_key

    def _key_exists(self, key: str) -> bool:
        """检查碎片是否存在（双保险）。

        client 不可用 / Redis 不可达 → 容错放行 True（白名单已守一道）。
        """
        try:
            client = self._storage._get_client()  # noqa: SLF001
        except Exception:
            return True
        if client is None:
            return True
        try:
            return bool(client.exists(key))
        except Exception:
            return True

    @staticmethod
    def _key_for_content(text: str) -> str:
        """根据文本推 Redis key（与 storage.store 一致的 sha256[:12] 算法）。"""
        import hashlib
        return f"memory:frag:{hashlib.sha256(text.encode()).hexdigest()[:12]}"

    @staticmethod
    def _tags_for(fact: Fact) -> str:
        """根据 fact.kind 派生 tags，保留 agent 注入由 storage.store 自己加。"""
        return f"kind:{fact.kind},pipeline:v2"

    # ------------------------------------------------------------------
    # 兜底：v1 路径
    # ------------------------------------------------------------------

    def _fallback_to_v1(self, turns: List[Turn], result: DrainResult, *, reason: str) -> DrainResult:
        """窗口级兜底：每条 user 走 v1（gate_fallback 或 storage.store 直存）。

        设计点：
          * 优先用 gate_fallback（封装了 decide() + store 的完整 v1 路径）
          * 没有 gate_fallback 时直接 storage.store（最原始兜底）
          * 不抛异常 —— 上层 sync_turn 已经返回，必须保证消息落地
        """
        result.fallback_v1 = True
        result.fallback_reason = reason
        logger.warning(
            "keepsake pipeline: fallback to v1 (reason=%s, turns=%d)",
            reason, len(turns),
        )
        for t in turns:
            if self._gate_fallback is not None:
                try:
                    self._gate_fallback(t.user, "turn_memory")
                    continue
                except Exception as e:
                    logger.debug("pipeline: gate_fallback raised: %s; trying raw store", e)
            # 最后兜底：直接存（不经过闸门 — 风险高，但保证不丢）
            try:
                self._storage.store(
                    text=t.user,
                    tags="conversation,fallback:v1",
                    category="turn_memory",
                    source="pipeline_v2_fallback",
                    fragment_type="memory",
                )
            except Exception as e:
                logger.warning("pipeline: raw store fallback failed: %s", e)
        return result


# ---------------------------------------------------------------------------
# 内部异常（不导出）
# ---------------------------------------------------------------------------

class _LLMFailure(Exception):
    """LLM 调用或 JSON 解析失败 —— 触发窗口级 v1 兜底。"""


class _BudgetExceeded(Exception):
    """窗口内 LLM 调用数超 max_calls_per_window —— 触发窗口级 v1 兜底。"""


# ---------------------------------------------------------------------------
# 公共导出
# ---------------------------------------------------------------------------

__all__ = [
    "Pipeline",
    "Turn",
    "Fact",
    "DrainResult",
    "DEFAULT_PIPELINE_CONFIG",
    "VALID_KINDS",
    "VOID_KEY",
]
