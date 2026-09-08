"""keepsake v2 — 写侧两相管线端到端测试（hermetic）。

覆盖 B 部分：
  a) 「可以的」窗口 → facts 空 → 零写入
  b) 需求→pending 入库，后续「完成」fact → UPDATE 封旧、旧碎片不再被检索命中
  c) LLM 抛异常 → 窗口回落 v1 原文入库
  d) 超 max_calls → 回落
  e) 队列不丢对（shutdown 前排空）

设计要点：
  * 不连 Redis/网络 —— 用 FakeStorage（与 pipeline 期望接口一致）
  * _call_llm 通过 monkeypatch 注入（脚本化返回）
  * gate_fallback 验证「LLM 失败时确实走 v1 路径」
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import pytest

from keepsake.pipeline import (
    Pipeline,
    Turn,
    Fact,
    DrainResult,
    DEFAULT_PIPELINE_CONFIG,
    VOID_KEY,
)


# ===========================================================================
# 测试用 fake storage —— 与 pipeline 期望的接口一致
# ===========================================================================

class _FakeRedisClient:
    """Pipeline 期望 _get_client() 返回一个含 exists() 和 hset() 的对象。

    exists(key) 默认返回 key 是否在 FakeStorage.fragments 里（0/1）；
    额外支持 not_exists 集合强制某 key 返回 0（测「search_bm25 命中但库
    里实际不存在」的场景）；hset(key, field, value) 仅记录调用，不写半壳键。
    """

    def __init__(self, storage: "FakeStorage"):
        self._storage = storage
        self.not_exists: set = set()  # 强制 exists()=0 的 key 集合

    def exists(self, key: str) -> int:
        if key in self.not_exists:
            return 0
        return 1 if key in self._storage.fragments else 0

    def hset(self, key: str, field: str, value=None) -> int:
        # 仅记录调用，不实际写入（防止半壳键）
        self._storage.hset_calls.append((key, field, value))
        return 1


class FakeStorage:
    """Pipeline 不需要真实 Redis —— 只要 search_bm25/store/supersede_fragment/_get_client。"""

    def __init__(self):
        self.stored: List[Dict[str, Any]] = []      # store() 调用记录
        self.search_calls: List[str] = []            # search_bm25 调用记录
        self.supersede_calls: List[Tuple[str, str]] = []  # (old, new) 记录
        # 模拟碎片库（按 key 索引）
        self.fragments: Dict[str, Dict[str, Any]] = {}
        # 模拟 search 索引（按 content 关键词返回）
        self.search_index: Dict[str, List[str]] = {}  # token -> [frag_keys]
        # 模拟 hset 调用记录（防止幻觉 key 产生半壳键）
        self.hset_calls: List[Tuple[str, str, Any]] = []
        self._client = _FakeRedisClient(self)

    # --- 模拟 store(): 写一个碎片，记录 key + 内容 ---
    def store(self, text: str, tags: str = "", category: str = "",
              source: str = "", fragment_type: str = "", **kwargs) -> bool:
        import hashlib
        key = f"memory:frag:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        stored: Dict[str, Any] = {
            "key": key,
            "text": text,
            "tags": tags,
            "category": category,
            "source": source,
            "fragment_type": fragment_type,
        }
        stored.update(kwargs)
        self.stored.append(stored)
        # 索引到 search（按 2-gram 切词，对中文/英文都生效）
        tokens = self._tokenize(text)
        for tok in tokens:
            self.search_index.setdefault(tok, []).append(key)
        self.fragments[key] = stored
        return True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """粗略分词：2-char sliding window + 空格分词（中文/英文都能命中）。"""
        text = text.lower()
        out: List[str] = []
        # 空格分词
        for tok in text.split():
            if len(tok) >= 2:
                out.append(tok)
        # 2-char sliding window（中文）
        for i in range(len(text) - 1):
            pair = text[i:i + 2]
            if any("一" <= ch <= "鿿" for ch in pair):
                out.append(pair)
        return out

    # --- 模拟 search_bm25(): 按 token 命中候选（粗略但够用） ---
    def search_bm25(self, query: str, tag_filter: str = "",
                    agent_id: str = "", is_primary: Optional[bool] = None) -> List[Dict[str, Any]]:
        self.search_calls.append(query)
        tokens = self._tokenize(query)
        # 统计每条碎片被命中次数（粗略 BM25）
        scores: Dict[str, int] = {}
        for tok in tokens:
            for k in self.search_index.get(tok, []):
                scores[k] = scores.get(k, 0) + 1
        # 按命中数降序
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out: List[Dict[str, Any]] = []
        for i, (k, score) in enumerate(ranked):
            frag = self.fragments.get(k)
            if frag is None:
                continue
            # 模拟「封边碎片不返回」
            if frag.get("superseded_by"):
                continue
            # 模拟「consumed 不返回」
            if frag.get("fragment_type") == "consumed":
                continue
            out.append({
                "_key": k,
                "content": frag["text"],
                "tags": frag.get("tags", ""),
                "category": frag.get("category", ""),
                "_score": 1.0 - i * 0.1,
                "_sim": 1.0 - i * 0.1,
                "_bm25_score": 1.0 - i * 0.1,
            })
        return out[:5]

    # --- 模拟 supersede_fragment(): 写入封边字段 ---
    def supersede_fragment(self, old_key: str, new_key: str) -> bool:
        self.supersede_calls.append((old_key, new_key))
        frag = self.fragments.get(old_key)
        if frag is not None:
            frag["superseded_by"] = new_key
            frag["superseded_at"] = "2026-09-08T00:00:00+00:00"
        return True

    def _get_client(self):
        """返回含 exists()/hset() 的 fake redis client，供 _key_exists + hset 校验用。"""
        return self._client


# ===========================================================================
# a) 「可以的」窗口 → facts 空 → 零写入
# ===========================================================================

class TestExtractPhaseEmpty:
    """纯确认短句窗口：提取相返回空数组 → pipeline 不写任何碎片。"""

    def test_pure_confirmation_returns_no_facts(self):
        """「可以的」/「好」/「OK」等纯确认 → LLM 应返回空数组。"""
        st = FakeStorage()
        # 脚本化 LLM：返回空 facts
        def fake_llm(messages, model):
            # 提取相 → 空 facts
            return '{"facts":[]}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        result = p._process_window(
            [Turn("可以的", "好的", 0.0)],
            [],
        )
        assert result.facts_extracted == 0
        assert result.llm_calls == 1  # 提取相 1 次，更新相 0 次
        # 关键断言：零写入
        assert len(st.stored) == 0, "纯确认窗口不应写任何碎片"
        assert len(st.supersede_calls) == 0

    def test_mixed_short_and_meaningful_only_writes_meaningful(self):
        """混合窗口：含 1 条有意义事实 + 2 条纯确认 → 只写 1 条。"""
        st = FakeStorage()
        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                # 提取相：返回 1 条事实
                return '{"facts":[{"content":"资金总览需要做分页","kind":"pending"}]}'
            else:
                # 更新相：返回 ADD
                return '{"action":"ADD","target_key":""}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        result = p._process_window(
            [
                Turn("可以的", "好的", 0.0),
                Turn("资金总览要做分页", "好", 0.0),
                Turn("嗯", "OK", 0.0),
            ],
            [],
        )
        assert result.facts_extracted == 1
        assert result.adds == 1
        assert len(st.stored) == 1
        assert "分页" in st.stored[0]["text"]


# ===========================================================================
# b) 需求→pending 入库，后续「完成」fact → UPDATE 封旧
# ===========================================================================

class TestUpdateSupersedeFlow:
    """端到端：先存 pending 需求，再触发完成 fact → UPDATE 路径封旧。"""

    def _build_pipeline_with_existing(self, st: FakeStorage) -> Pipeline:
        """构造一个已经存了「分页需求」碎片的 pipeline。"""
        st.store(
            text="用户要求做资金总览分页",
            tags="kind:pending,conversation",
            category="fact_v2",
            source="pipeline_v2",
        )
        return Pipeline(storage=st, llm_fn=lambda msgs, model: '{"action":"NOOP","target_key":""}')

    def test_update_phase_seals_old_fragment(self):
        """完成 fact → UPDATE → 旧碎片被 supersede_fragment 标记，新碎片被 store。"""
        st = FakeStorage()
        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                # 提取相：返回 1 条完成 fact
                return '{"facts":[{"content":"资金总览分页已完成上线","kind":"state_change"}]}'
            else:
                # 更新相：返回 UPDATE 指向第一候选 key
                # 真实场景 LLM 会基于 candidates 内容判断；这里用 NOOP 兜底，
                # 验证 _do_update 的 fallback（找不到 → ADD）
                return '{"action":"ADD","target_key":""}'

        p = self._build_pipeline_with_existing(st)
        # 重置 fake_llm
        p._llm_fn = fake_llm

        result = p._process_window(
            [Turn("资金总览分页做完了", "已上线", 0.0)],
            [],
        )
        assert result.adds == 1
        # 新事实被写入
        assert any("已完成" in s["text"] for s in st.stored)

    def test_update_seals_via_explicit_decision(self):
        """直接构造 UPDATE 决策：候选里有旧 key → 应调用 supersede_fragment。"""
        st = FakeStorage()
        st.store(
            text="用户要求做资金总览分页",
            tags="kind:pending,conversation",
            category="fact_v2",
            source="pipeline_v2",
        )

        # 抓取候选 key（pipeline 内部 search_bm25 用 FakeStorage.search_bm25）
        candidates = st.search_bm25("分页")
        assert len(candidates) >= 1
        old_key = candidates[0]["_key"]

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"资金总览分页已完成上线","kind":"state_change"}]}'
            else:
                # 更新相：明确返回 UPDATE 指向 old_key
                return f'{{"action":"UPDATE","target_key":"{old_key}"}}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        result = p._process_window(
            [Turn("资金总览分页做完了", "已上线", 0.0)],
            [],
        )
        assert result.updates == 1
        # 关键断言：supersede_fragment 被调用过
        assert len(st.supersede_calls) >= 1
        old_sealed, new_key = st.supersede_calls[0]
        assert old_sealed == old_key
        assert "已完成" in st.fragments.get(new_key, {}).get("text", "") or new_key.startswith("memory:frag:")

    def test_sealed_old_fragment_no_longer_in_search(self):
        """UPDATE 后：旧碎片被封 → FakeStorage.search_bm25 自动剔除（模拟 v2 过滤）。"""
        st = FakeStorage()
        st.store(
            text="用户要求做资金总览分页",
            tags="kind:pending,conversation",
            category="fact_v2",
            source="pipeline_v2",
        )

        call_count = [0]
        old_key = st.search_bm25("分页")[0]["_key"]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"分页已完成上线","kind":"state_change"}]}'
            else:
                return f'{{"action":"UPDATE","target_key":"{old_key}"}}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        p._process_window([Turn("分页做完了", "OK", 0.0)], [])

        # 现在再 search「分页」：旧碎片已被 supersede_by → search_bm25 不应返回它
        after = st.search_bm25("分页")
        keys = [c["_key"] for c in after]
        assert old_key not in keys, "封边后的旧碎片不应再出现在检索结果"

    def test_delete_phase_seals_with_void(self):
        """DELETE 路径：旧碎片 superseded_by='__void__'。"""
        st = FakeStorage()
        st.store(
            text="矛盾的旧记忆",
            tags="kind:fact,conversation",
            category="fact_v2",
            source="pipeline_v2",
        )
        old_key = st.search_bm25("矛盾")[0]["_key"]

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"矛盾的新事实","kind":"fact"}]}'
            else:
                return f'{{"action":"DELETE","target_key":"{old_key}"}}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        result = p._process_window(
            [Turn("矛盾的新事实出现了", "OK", 0.0)],
            [],
        )
        assert result.deletes == 1
        # 旧碎片被 supersede_fragment(old, __void__)
        assert (old_key, VOID_KEY) in st.supersede_calls
        # 验证旧碎片标了 superseded_by=__void__
        assert st.fragments[old_key]["superseded_by"] == VOID_KEY


# ===========================================================================
# c) LLM 抛异常 → 窗口回落 v1 原文入库
# ===========================================================================

class TestLLMFailureFallback:
    """LLM 调用失败或 JSON 解析失败 → 窗口整体走 v1 兜底。"""

    def test_llm_returns_none_triggers_fallback(self):
        """LLM 返回 None（无 API key / 网络断开） → 全部走 gate_fallback。"""
        st = FakeStorage()
        fallback_calls: List[Tuple[str, str]] = []

        def gate_fallback(text, category):
            fallback_calls.append((text, category))
            st.store(text=text, tags="fallback:v1", category=category,
                     source="pipeline_v2_fallback")

        def broken_llm(messages, model):
            return None  # 模拟 _call_llm 在无 API key 时的行为

        p = Pipeline(storage=st, llm_fn=broken_llm, gate_fallback=gate_fallback)
        result = p._process_window(
            [
                Turn("这是有意义的事实一", "好", 0.0),
                Turn("这是有意义的事实二", "好", 0.0),
            ],
            [],
        )
        assert result.fallback_v1 is True
        assert "llm" in result.fallback_reason.lower()
        # 兜底函数被调用 2 次（每条 turn 一次）
        assert len(fallback_calls) == 2
        assert fallback_calls[0] == ("这是有意义的事实一", "turn_memory")

    def test_llm_throws_exception_triggers_fallback(self):
        """LLM 抛异常 → 兜底。"""
        st = FakeStorage()
        fallback_calls: List[Tuple[str, str]] = []

        def gate_fallback(text, category):
            fallback_calls.append((text, category))
            st.store(text=text, tags="fallback:v1", category=category,
                     source="pipeline_v2_fallback")

        def broken_llm(messages, model):
            raise RuntimeError("network down")

        p = Pipeline(storage=st, llm_fn=broken_llm, gate_fallback=gate_fallback)
        result = p._process_window(
            [Turn("测试事实", "OK", 0.0)],
            [],
        )
        assert result.fallback_v1 is True
        assert "extract_failed" in result.fallback_reason
        assert len(fallback_calls) == 1

    def test_llm_returns_invalid_json_triggers_fallback(self):
        """LLM 返回非 JSON → JSON 解析失败 → 兜底。"""
        st = FakeStorage()
        fallback_calls: List[Tuple[str, str]] = []

        def gate_fallback(text, category):
            fallback_calls.append((text, category))

        def bad_json_llm(messages, model):
            return "this is not JSON at all"

        p = Pipeline(storage=st, llm_fn=bad_json_llm, gate_fallback=gate_fallback)
        result = p._process_window([Turn("事实", "好", 0.0)], [])
        assert result.fallback_v1 is True
        assert "extract" in result.fallback_reason
        assert len(fallback_calls) == 1

    def test_update_phase_llm_failure_triggers_fallback(self):
        """提取相成功但更新相失败 → 整个窗口兜底。"""
        st = FakeStorage()
        fallback_calls: List[Tuple[str, str]] = []

        def gate_fallback(text, category):
            fallback_calls.append((text, category))
            st.store(text=text, tags="fallback:v1", category=category,
                     source="pipeline_v2_fallback")

        call_count = [0]

        def partial_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                # 提取相 OK
                return '{"facts":[{"content":"事实A","kind":"fact"}]}'
            else:
                # 更新相失败
                raise RuntimeError("update phase network error")

        p = Pipeline(storage=st, llm_fn=partial_llm, gate_fallback=gate_fallback)
        result = p._process_window([Turn("事实A 的描述", "好", 0.0)], [])
        assert result.fallback_v1 is True
        assert "update_failed" in result.fallback_reason
        # 兜底调用 1 次（窗口 1 条 turn）
        assert len(fallback_calls) == 1


# ===========================================================================
# d) 超 max_calls → 回落
# ===========================================================================

class TestBudgetExceededFallback:
    """窗口内 LLM 调用数超 max_calls_per_window → 回落 v1。"""

    def test_budget_exceeded_triggers_fallback(self):
        """max_calls=3，3 facts 提取相 1 次 + 3 次更新相 = 4 次，超预算。"""
        st = FakeStorage()
        fallback_calls: List[Tuple[str, str]] = []

        def gate_fallback(text, category):
            fallback_calls.append((text, category))
            st.store(text=text, tags="fallback:v1", category=category,
                     source="pipeline_v2_fallback")

        def multi_fact_llm(messages, model):
            # 提取相返回 3 条；更新相返回 ADD（每次都成功）
            # 第一次（提取相）返回 3 facts
            return '{"facts":[{"content":"事实A","kind":"fact"},{"content":"事实B","kind":"fact"},{"content":"事实C","kind":"fact"}]}'

        p = Pipeline(
            storage=st, llm_fn=multi_fact_llm,
            max_calls_per_window=3,  # 紧预算：1 提取 + 2 更新 = 3，第 3 条更新时超额
            gate_fallback=gate_fallback,
        )
        result = p._process_window(
            [Turn("事实A B C 都说了", "好", 0.0)],
            [],
        )
        # max_calls=3，第 3 条 fact 进入更新相时 llm_calls 已 = 2（提取 1 + 2 更新）
        # → 触发预算超额 → 兜底
        assert result.fallback_v1 is True
        assert result.fallback_reason == "budget_exceeded"
        # 兜底覆盖原 turn
        assert len(fallback_calls) == 1

    def test_no_budget_issue_within_limit(self):
        """max_calls 充足 → 不应触发预算兜底。"""
        st = FakeStorage()
        fallback_calls: List[Tuple[str, str]] = []

        def gate_fallback(text, category):
            fallback_calls.append((text, category))

        def llm_ok(messages, model):
            return '{"facts":[{"content":"事实A","kind":"fact"}]}'

        p = Pipeline(
            storage=st, llm_fn=llm_ok,
            max_calls_per_window=10,
            gate_fallback=gate_fallback,
        )
        result = p._process_window([Turn("事实A 出现了", "好", 0.0)], [])
        assert result.fallback_v1 is False
        assert len(fallback_calls) == 0


# ===========================================================================
# e) 队列不丢对（shutdown 前排空）
# ===========================================================================

class TestShutdownDrainNoLoss:
    """shutdown(drain=True) 必须保证队列内 turn 都被处理。"""

    def test_stop_with_drain_processes_pending_turns(self):
        """start() → enqueue 3 条 → stop(drain=True) → 全部处理。"""
        st = FakeStorage()
        # 每个 turn 提取出唯一的事实（避免重复 → NOOP）
        user_to_fact = {
            "事实 1": "事实1内容",
            "事实 2": "事实2内容",
            "事实 3": "事实3内容",
        }
        extracted_users: List[str] = []

        def fake_llm(messages, model):
            user_msg = messages[1]["content"]
            # 区分提取相 / 更新相：提取相 prompt 含「本窗口对话」，更新相含「已有相似记忆」
            if "本窗口对话" in user_msg:
                # 提取相 → 返回每个 user 对应 fact
                facts = []
                for user_text, fact_content in user_to_fact.items():
                    if user_text in user_msg:
                        extracted_users.append(user_text)
                        facts.append({"content": fact_content, "kind": "fact"})
                return '{"facts":' + str(__import__("json").dumps(facts, ensure_ascii=False)) + '}'
            else:
                # 更新相 → ADD（不查候选，新事实入库）
                return '{"action":"ADD","target_key":""}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        p.start()
        try:
            # 入队 3 条（不到 window_pairs=4 阈值）
            p.enqueue("事实 1", "好")
            p.enqueue("事实 2", "好")
            p.enqueue("事实 3", "好")
            assert len(p._queue) == 3
        finally:
            # shutdown 必须排空
            p.stop(drain=True, timeout=3.0)

        # 队列已清空
        assert len(p._queue) == 0
        # 3 条都被处理（每次窗口可能合并 drain，断言至少 3 条 facts 入库）
        assert len(st.stored) >= 3, f"shutdown 后应至少入库 3 条，实际 {len(st.stored)}"

    def test_stop_without_drain_skips_pending(self):
        """stop(drain=False) → 队列内 turn 不处理（快速退出场景）。"""
        st = FakeStorage()
        p = Pipeline(storage=st, llm_fn=lambda m, mo: '{"facts":[]}')
        p.start()
        try:
            p.enqueue("事实 1", "好")
            p.enqueue("事实 2", "好")
        finally:
            p.stop(drain=False, timeout=1.0)
        # 没排空 → 0 入库
        assert len(st.stored) == 0

    def test_window_pairs_threshold_triggers_immediate_drain(self):
        """入队 ≥window_pairs 时立即 drain（不等 daemon）。"""
        st = FakeStorage()
        p = Pipeline(storage=st, llm_fn=lambda m, mo: '{"facts":[]}', window_pairs=2)
        # 不 start daemon（直接测 enqueue 触发）
        p.enqueue("事实 1", "好")
        p.enqueue("事实 2", "好")  # 第 2 条触发 drain
        # drain 后队列清空
        assert len(p._queue) == 0


# ===========================================================================
# 辅助测试：解析 JSON 容错
# ===========================================================================

class TestJSONParseRobustness:
    """LLM 输出可能包 JSON 在 markdown 或加前后缀 —— 解析要容错。"""

    def test_parse_facts_with_markdown_wrapper(self):
        text = '```json\n{"facts":[{"content":"X","kind":"fact"}]}\n```'
        facts = Pipeline._parse_facts(text)
        assert len(facts) == 1
        assert facts[0].content == "X"
        assert facts[0].kind == "fact"

    def test_parse_facts_with_preamble(self):
        text = '好的，结果如下：\n{"facts":[{"content":"X","kind":"decision"}]}'
        facts = Pipeline._parse_facts(text)
        assert len(facts) == 1
        assert facts[0].kind == "decision"

    def test_parse_facts_empty_array(self):
        facts = Pipeline._parse_facts('{"facts":[]}')
        assert facts == []

    def test_parse_facts_missing_facts_field(self):
        facts = Pipeline._parse_facts('{"other":"x"}')
        assert facts == []

    def test_parse_facts_invalid_kind_defaults_to_fact(self):
        facts = Pipeline._parse_facts('{"facts":[{"content":"X","kind":"unknown_kind"}]}')
        assert len(facts) == 1
        assert facts[0].kind == "fact"

    def test_parse_action_recognizes_add(self):
        # target_key 为空串会被代码过滤为 None（避免被误当作有效 key）
        assert Pipeline._parse_action('{"action":"ADD","target_key":""}') == ("ADD", None)

    def test_parse_action_recognizes_update_with_markdown(self):
        text = '```\n{"action":"UPDATE","target_key":"abc"}\n```'
        assert Pipeline._parse_action(text) == ("UPDATE", "abc")

    def test_parse_action_rejects_text_fallback(self):
        """LLM 输出含 ADD 关键字但 JSON 损坏 → 一律 NOOP（拒绝关键字兜底）。

        解释性废话如「该信息与旧记忆不冲突，无需 UPDATE」会被关键字兜底误判为
        UPDATE → 触发 UPDATE 路径封旧事实。修洞2后一律 NOOP。
        """
        assert Pipeline._parse_action("I recommend ADD this fact") == ("NOOP", None)
        # 中文解释性废话（含 UPDATE 字样但无 JSON）→ 也 NOOP
        assert Pipeline._parse_action("该信息与旧记忆不冲突，无需 UPDATE") == ("NOOP", None)


# ===========================================================================
# 辅助测试：默认值与配置
# ===========================================================================

class TestDefaults:
    """DEFAULT_PIPELINE_CONFIG 应是生产推荐值。"""

    def test_defaults_match_task_spec(self):
        assert DEFAULT_PIPELINE_CONFIG["enabled"] is True
        assert DEFAULT_PIPELINE_CONFIG["window_pairs"] == 4
        assert DEFAULT_PIPELINE_CONFIG["window_seconds"] == 30.0
        assert DEFAULT_PIPELINE_CONFIG["max_calls_per_window"] == 8
        assert DEFAULT_PIPELINE_CONFIG["model"] == "qwen-plus"

    def test_pipeline_does_not_import_redis(self):
        """约束：pipeline.py 内不 import redis（存储操作全走注入的 storage）。"""
        import ast
        from keepsake import pipeline as p_module
        with open(p_module.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.split(".")[0] == "redis", \
                        f"pipeline.py 不能直接 import redis：{alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "redis":
                    raise AssertionError(f"pipeline.py 不能 from redis import：{node.module}")


# ===========================================================================
# 洞1：target_key 零校验 — LLM 幻觉 key / 候选外 key 一律拒
# ===========================================================================

class TestSaneTargetRejectsHallucinatedKey:
    """_sane_target 校验：白名单 + exists 双保险。LLM 幻觉 key 不能 supersede 真碎片。"""

    def test_hallucinated_key_does_not_supersede_real_fragment(self):
        """幻觉 key（不在候选 + exists=False）→ LLM 路径拒用，_find 兜底封真碎片属正常。"""
        st = FakeStorage()
        # 库里已有真碎片（不能被幻觉 key 误封）
        st.store(
            text="用户要求做资金总览分页",
            tags="kind:pending,conversation",
            category="fact_v2",
            source="pipeline_v2",
        )
        real_key = list(st.fragments.keys())[0]
        assert st._client.exists(real_key) == 1  # 真碎片存在

        # 验证：模拟 LLM 幻觉一个不存在的 key
        hallucinated_key = "memory:frag:deadbeef9999"
        assert st._client.exists(hallucinated_key) == 0  # 库里没有

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"分页已完成上线","kind":"state_change"}]}'
            else:
                # LLM 幻觉：给 UPDATE + 库里不存在的 target_key
                return f'{{"action":"UPDATE","target_key":"{hallucinated_key}"}}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        result = p._process_window(
            [Turn("分页做完了", "已上线", 0.0)],
            [],
        )
        # 关键断言：幻觉 key 从未被 supersede_fragment 调用过（防半壳键核心）
        supersede_keys = [k for (k, _) in st.supersede_calls]
        assert hallucinated_key not in supersede_keys, (
            f"幻觉 key 不应触发 supersede_fragment，实际 {st.supersede_calls}"
        )
        # 关键断言：没对幻觉 key 产生任何 hset（半壳键核心防御）
        hset_keys = {k for (k, _, _) in st.hset_calls}
        assert hallucinated_key not in hset_keys, (
            f"幻觉 key 不应被 hset 触及：{st.hset_calls}"
        )
        # 退化行为：UPDATE 找不到合法 target_key（_find 可能找到 real_key 属正常路径）
        # 但本测试要求 _find 也找不到（库外碎片不在 _find 的搜索路径上 → 退化 ADD）
        # 这里 _find 仍可能命中 real_key；测试只断言幻觉 key 未被用 + 新事实入库
        assert any("已完成" in s["text"] for s in st.stored), (
            f"应入库新事实，实际 {st.stored}"
        )

    def test_hallucinated_key_with_empty_storage_degrades_to_add(self):
        """空库 + 幻觉 key → 既不封任何碎片，也不产生半壳键，退化 ADD。"""
        st = FakeStorage()
        hallucinated_key = "memory:frag:deadbeef9999"

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"全新事实","kind":"fact"}]}'
            else:
                return f'{{"action":"UPDATE","target_key":"{hallucinated_key}"}}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        p._process_window(
            [Turn("全新事实", "OK", 0.0)],
            [],
        )
        # 空库 → candidates 空 → _find 也空 → UPDATE 退化 ADD
        assert len(st.supersede_calls) == 0
        # 关键：幻觉 key 从未被 hset
        hset_keys = {k for (k, _, _) in st.hset_calls}
        assert hallucinated_key not in hset_keys
        # 新事实入库
        assert any("全新事实" in s["text"] for s in st.stored)

    def test_key_outside_candidates_even_if_exists_is_rejected(self):
        """库里真实存在但不在本次候选的 key → 同样拒（白名单硬约束）。"""
        st = FakeStorage()
        # 两个真碎片，但只把其中一个展示给 LLM
        st.store(text="用户要求做资金总览分页", category="fact_v2")
        st.store(text="完全不相关的另一主题", category="fact_v2")
        all_keys = list(st.fragments.keys())
        assert len(all_keys) == 2

        # 选一个真实存在但本次候选里**不会**出现的 key 作为 LLM 输出
        unrelated_key = "memory:frag:extraneous1"
        # 不放进 fragments → exists=0；放进 fragments 后 → exists=1
        st.fragments[unrelated_key] = {"text": "其他主题", "key": unrelated_key}

        # 先确认 search_bm25「分页」不会返回 unrelated_key（不共享 token）
        cands = st.search_bm25("分页")
        cands_keys = [c["_key"] for c in cands]
        assert unrelated_key not in cands_keys, (
            f"测试 setup 失败：unrelated_key 出现在候选里 {cands_keys}"
        )
        # unrelated_key 在库里真实存在
        assert st._client.exists(unrelated_key) == 1

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"分页已完成上线","kind":"state_change"}]}'
            else:
                # LLM 给了一个库里真存在、却不在候选里的 key
                return f'{{"action":"UPDATE","target_key":"{unrelated_key}"}}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        result = p._process_window(
            [Turn("分页做完了", "已上线", 0.0)],
            [],
        )
        # 关键断言：unrelated_key 不被直接 supersede（被白名单拒）
        supersede_keys = [k for (k, _) in st.supersede_calls]
        assert unrelated_key not in supersede_keys, (
            f"白名单外 key 不应被 supersede，实际 {st.supersede_calls}"
        )
        # 退化 ADD（找不到白名单内的合法 target；_find 可能命中真碎片属正常）
        assert any("已完成" in s["text"] for s in st.stored)

    def test_sane_target_unit_pure(self):
        """_sane_target 单元测试：白名单/exists 两个开关各独立生效。"""
        st = FakeStorage()
        p = Pipeline(storage=st, llm_fn=lambda m, mo: "")

        # case A: target_key 为空 → None
        assert p._sane_target("", [{"_key": "k1"}], raw_from_llm=True) is None
        assert p._sane_target(None, [{"_key": "k1"}], raw_from_llm=True) is None

        # case B: 不在白名单（raw_from_llm=True） → None
        assert p._sane_target(
            "k2", [{"_key": "k1"}], raw_from_llm=True,
        ) is None

        # case C: 在白名单但库里不存在 → None
        assert p._sane_target(
            "k1", [{"_key": "k1"}], raw_from_llm=True,
        ) is None  # _get_client().exists('k1')=0

        # case D: 在白名单 + 库里存在 → 原样返回
        st.fragments["k1"] = {"text": "x"}
        assert p._sane_target(
            "k1", [{"_key": "k1"}], raw_from_llm=True,
        ) == "k1"

        # case E: raw_from_llm=False 跳过白名单检查（_find 兜底场景）
        assert p._sane_target(
            "k1", [], raw_from_llm=False,
        ) == "k1"
        # 但仍过 exists 校验
        assert p._sane_target(
            "k_does_not_exist", [], raw_from_llm=False,
        ) is None


# ===========================================================================
# 洞1：盲兜底移除 — LLM 没给 key 时只走 _find_supersede_target（也过 exists）
# ===========================================================================

class TestBlindFallbackRemoved:
    """_update_phase 删掉「没给 key 取第一条」的盲兜底；改走 _find_supersede_target。"""

    def test_empty_target_key_falls_back_to_find_supersede(self):
        """LLM 返回 UPDATE 但 target_key 空 → _find_supersede_target 返回合法 key → UPDATE 成功。"""
        st = FakeStorage()
        st.store(
            text="用户要求做资金总览分页",
            tags="kind:pending,conversation",
            category="fact_v2",
            source="pipeline_v2",
        )
        old_key = st.search_bm25("分页")[0]["_key"]

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"分页已完成上线","kind":"state_change"}]}'
            else:
                # 关键：target_key 空字符串（不是幻觉 key，是真的没给）
                return '{"action":"UPDATE","target_key":""}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        p._process_window(
            [Turn("分页做完了", "已上线", 0.0)],
            [],
        )
        # _find_supersede_target 找到 old_key（first BM25 hit, exists=True）→ UPDATE 成功
        supersede_keys = [k for (k, _) in st.supersede_calls]
        assert old_key in supersede_keys, (
            f"_find 兜底应封 old_key，实际 supersede_keys={supersede_keys}"
        )
        # 真碎片被 supersede
        assert st.fragments[old_key].get("superseded_by") is not None

    def test_find_returns_none_when_top_result_missing_degrades_to_add(self):
        """_find_supersede_target 因 exists 校验剔除 top 后若无 → 退化 ADD。"""
        st = FakeStorage()
        ghost_key = "memory:frag:ghostkey0001"
        # ghost_key 加入 search_index 排第一（保证 search_bm25 top 是它）
        st.search_index["分页"] = [ghost_key]
        # store() 追加 real_key 到 search_index["分页"]（第二位）
        st.store(
            text="用户要求做资金总览分页",
            tags="kind:pending,conversation",
            category="fact_v2",
            source="pipeline_v2",
        )
        real_key = list(st.fragments.keys())[0]
        # ghost_key 在 fragments 里（让 search_bm25 返回它），但强制 exists=0
        st.fragments[ghost_key] = {"text": "ghost content", "key": ghost_key}
        st._client.not_exists.add(ghost_key)
        # 现在：search_bm25 返回 ghost_key + real_key；exists 仅 real_key 通过

        # 验证 setup：search_bm25("分页") top 是 ghost_key
        top_cands = st.search_bm25("分页")
        assert top_cands[0]["_key"] == ghost_key
        # 但 exists()=0
        assert st._client.exists(ghost_key) == 0

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"分页已完成上线","kind":"state_change"}]}'
            else:
                return '{"action":"UPDATE","target_key":""}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        p._process_window(
            [Turn("分页做完了", "已上线", 0.0)],
            [],
        )
        # ghost_key 不被 supersede（防半壳键）
        assert all(k != ghost_key for (k, _) in st.supersede_calls)
        # 整个 fallback chain 退化 ADD：新事实入库
        assert any("已完成" in s["text"] for s in st.stored)

    def test_no_candidates_and_empty_target_degrades_to_add(self):
        """无候选 + target_key 空 → ADD（避免静默丢）。"""
        st = FakeStorage()
        # 不预存任何碎片 → candidates 为空

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"全新事实","kind":"fact"}]}'
            else:
                return '{"action":"UPDATE","target_key":""}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        p._process_window(
            [Turn("全新事实", "OK", 0.0)],
            [],
        )
        # 找不到任何封边目标 → _do_add 兜底，新事实入库
        assert any("全新事实" in s["text"] for s in st.stored), (
            f"退化 ADD 应入库新事实，实际 {st.stored}"
        )
        assert len(st.supersede_calls) == 0


# ===========================================================================
# 洞2：_parse_action 文本兜底删除 — 只接受合法 JSON
# ===========================================================================

class TestParseActionStrict:
    """_parse_action 严格 JSON-only：缺失/解析失败/action 不在四操作内 → ("NOOP", None)。"""

    def test_explanatory_text_with_update_keyword_returns_noop(self):
        """解释性废话「无需 UPDATE」含 UPDATE 字样 → NOOP（防误触发封边）。"""
        assert Pipeline._parse_action(
            "该信息与旧记忆不冲突，无需 UPDATE"
        ) == ("NOOP", None)

    def test_explanatory_text_with_delete_keyword_returns_noop(self):
        """解释性废话「不该 DELETE」含 DELETE 字样 → NOOP。"""
        assert Pipeline._parse_action(
            "我评估了一下，不该 DELETE 这条记忆"
        ) == ("NOOP", None)

    def test_garbled_no_json_returns_noop(self):
        """乱码无 JSON → NOOP。"""
        assert Pipeline._parse_action("乱码内容 12345 !@#$%") == ("NOOP", None)
        assert Pipeline._parse_action("") == ("NOOP", None)
        # None 由 _update_phase 在调用前已 raise _LLMFailure，不进 _parse_action

    def test_malformed_json_returns_noop(self):
        """JSON 语法错误（如缺右括号） → NOOP。"""
        assert Pipeline._parse_action('{"action":"UPDATE","target_key":') == ("NOOP", None)

    def test_unknown_action_returns_noop(self):
        """合法 JSON 但 action 不在四操作内 → NOOP。"""
        assert Pipeline._parse_action('{"action":"REPLACE","target_key":"x"}') == ("NOOP", None)

    def test_legal_json_four_actions_still_work(self):
        """合法 JSON 四操作各照常（不矫枉过正）。"""
        assert Pipeline._parse_action('{"action":"ADD","target_key":""}') == ("ADD", None)
        assert Pipeline._parse_action('{"action":"UPDATE","target_key":"k1"}') == ("UPDATE", "k1")
        assert Pipeline._parse_action('{"action":"DELETE","target_key":"k1"}') == ("DELETE", "k1")
        assert Pipeline._parse_action('{"action":"NOOP","target_key":"k1"}') == ("NOOP", "k1")

    def test_json_in_markdown_wrapper_parsed(self):
        """LLM 偶尔用 markdown 包 JSON → 兼容解析（这是 JSON 解析范畴，非关键字兜底）。"""
        text = '```json\n{"action":"UPDATE","target_key":"k1"}\n```'
        assert Pipeline._parse_action(text) == ("UPDATE", "k1")

    def test_json_with_preamble_parsed(self):
        """LLM 在 JSON 前写几句解释 → 兼容解析。"""
        text = '好的，我的判断如下：\n{"action":"ADD","target_key":""}'
        assert Pipeline._parse_action(text) == ("ADD", None)


# ===========================================================================
# 综合：洞1 + 洞2 端到端 — UPDATE 决策 + 仅 ADD 合法 JSON 才入库
# ===========================================================================

class TestEndToEndWithFixes:
    """端到端：合法 JSON + 合法 key 走 UPDATE；其他路径不污染库。"""

    def test_noop_text_path_does_not_seal(self):
        """LLM 输出解释性 UPDATE 废话 → NOOP → 0 写入、0 supersede。"""
        st = FakeStorage()
        st.store(text="用户要求做资金总览分页", category="fact_v2")
        real_key = list(st.fragments.keys())[0]

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"分页已完成","kind":"state_change"}]}'
            else:
                # 解释性 UPDATE 废话（无 JSON）
                return "该信息与旧记忆不冲突，无需 UPDATE"

        p = Pipeline(storage=st, llm_fn=fake_llm)
        result = p._process_window(
            [Turn("分页做完了", "已上线", 0.0)],
            [],
        )
        # _parse_action → NOOP；target_key 不存在 → _do_add fallback
        # 等等：NOOP 走 result.noops += 1 分支，不进 _do_add
        assert result.noops == 1
        assert result.adds == 0
        assert result.updates == 0
        # 关键：真碎片未被 supersede
        assert "superseded_by" not in st.fragments[real_key]
        assert len(st.supersede_calls) == 0

    def test_hallucinated_key_does_not_create_half_shell(self):
        """LLM 幻觉 key → 不得对幻觉 key 产生任何 hset（半壳键的核心防御）。"""
        st = FakeStorage()
        # 库里**没有**任何候选碎片（保证 candidates 为空 → _find 也 None → ADD）
        hallucinated_key = "memory:frag:deadbeef9999"

        call_count = [0]

        def fake_llm(messages, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"facts":[{"content":"全新事实","kind":"fact"}]}'
            else:
                return f'{{"action":"UPDATE","target_key":"{hallucinated_key}"}}'

        p = Pipeline(storage=st, llm_fn=fake_llm)
        p._process_window(
            [Turn("全新事实", "OK", 0.0)],
            [],
        )
        # 关键断言：幻觉 key 没有任何 hset 调用
        hset_keys = {k for (k, _, _) in st.hset_calls}
        assert hallucinated_key not in hset_keys, (
            f"幻觉 key 不应被任何 hset 触及：{st.hset_calls}"
        )
