"""写闸门（Ingest Gate）— store 之前的纯函数裁决中间件。

# 为什么做（2026-09-08 事故复盘）

Keepsake 当前「零过滤直存」让几类内容污染召回：

  * CONTEXT COMPACTION 摘要（数千字、含 30+ 过期任务快照）每轮被 store，
    下次检索被注入后把 7 月早已完成的需求当待办复活。
  * builtin MEMORY.md 条目经 on_memory_write 同步进碎片库，与 Hermes 本身
    双份召回。
  * 纯确认短句（"可以的"×15）sim≈0 仍被注入。

# 设计

`decide(text, category, existing_meta=None) -> IngestDecision`

  纯函数，零副作用、零 Redis 依赖，单测可直跑。
  规则按序短路 R1-R7：

    R1 compaction 摘要     → reject / reason="compaction"
    R2 超长 (len > 2000)    → reject / reason="oversize"
    R3 纯确认 / 短状态问句   → reject / reason="low_signal_short"
    R4 纯粘贴 (无字母/汉字/数字) → reject / reason="paste"
    R5 builtin MEMORY.md 同步 → reject / reason="builtin_dup"
    R6 同 hash 已存在 + turn_memory/conversation → update_state / reason="state_only"
    R7 其余                 → store / reason=""

R6 关键闭环：只刷新 updated_at/count，绝不覆盖原 content。
防「[会话摘要] 完成:X」被后来 compaction 残句毁掉终态。

# 调用点

  * KeepsakeProvider.sync_turn()   — store 前 decide() 裁决
  * KeepsakeProvider.on_memory_write() — 直接 return（双份入库关闭）
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


# ---------------------------------------------------------------------------
# 默认配置（与 KeepsakeProvider._resolve_config 对齐）
# ---------------------------------------------------------------------------

DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_len": 2000,
}


# ---------------------------------------------------------------------------
# 决策值
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestDecision:
    """decide() 的返回值。

    action: "store"          — 落库（走原 storage.store 逻辑）
            "update_state"   — 仅刷新 updated_at/count，不覆盖 content
            "reject"         — 拒收，调用方直接 return
    reason: 空串（store）或 R1-R6 的 reason 标识字符串。
    """

    action: str
    reason: str = ""


# ---------------------------------------------------------------------------
# R3 黑名单 + 状态问句正则（固定写全，禁止自作主张扩）
# ---------------------------------------------------------------------------

_CONFIRM_BLACKLIST = frozenset({
    "可以的", "可以", "好", "好的", "行", "行的", "嗯", "嗯嗯", "哦", "噢",
    "ok", "收到", "明白", "知道了", "懂了", "是的", "对", "对的",
    "没事了", "继续", "继续吧", "没有了", "不用", "不用了", "算了",
    "试试", "要的", "要", "加", "搞", "修", "部署", "提交", "合并",
    "看看", "停", "暂停", "恢复", "重试", "来吧", "行吧", "不用管",
    "先不管", "先不动", "随便", "都行",
})

# 状态问句：以这些前缀开头，允许尾部 0-1 个问号（全角或半角）
_STATUS_QUESTION_RE = re.compile(
    r"^(怎么样了|咋样了|目前咋样了|目前是什么情况|目前情况|现在呢|进展呢|结果呢|到哪了|弄好了吗|改好了吗|做完了吗|好了吗|完成了吗|还有吗|然后呢)[？?]?$"
)


# ---------------------------------------------------------------------------
# 工具：剥离空白/标点/符号/控制字符，仅保留字母与汉字/数字
# ---------------------------------------------------------------------------

def _strip_text(text: str) -> str:
    """剥离空白/标点/符号/控制字符，仅保留字母（L*）、数字（N*）、汉字（Lo）。

    R3 用：剥离后再做黑名单与状态问句匹配；emoji（多数归 So）一并剥掉。
    """
    out = []
    for ch in text:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N"):
            out.append(ch)
    return "".join(out).lower()


def _is_pure_paste(text: str) -> bool:
    """R4：纯粘贴判定 — 文本中无任何字母/汉字/数字字符。

    例：「==========」「------------」「////////////」整段 → reject。
    注：真正的 URL 多数含字母（https 等），不会命中 R4，会被 R3 长度阈值兜底。
    """
    for ch in text:
        # CJK 基本汉字 + Ext A
        if "一" <= ch <= "鿿":
            return False
        if "㐀" <= ch <= "䶿":
            return False
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N"):
            return False
    return True


# ---------------------------------------------------------------------------
# 入口：纯函数裁决
# ---------------------------------------------------------------------------

def decide(
    text: str,
    category: str,
    existing_meta: Optional[Dict[str, Any]] = None,
    gate_cfg: Optional[Dict[str, Any]] = None,
) -> IngestDecision:
    """写闸门裁决函数（纯函数，无副作用、无 Redis 依赖）。

    参数:
        text: 待写入的文本（建议调用方先 strip）。
        category: 调用方传入的 category（"turn_memory" / "memory_tool" / ...）。
        existing_meta: 若同内容 hash 已在碎片库，给出至少含 "key" 字段的元数据；
                       R6 用。None 或空 dict 表示无既有碎片。
        gate_cfg: 覆盖默认配置；None 表示用 DEFAULT_GATE_CONFIG。

    返回:
        IngestDecision（按 R1-R7 顺序短路）。
    """
    cfg = dict(DEFAULT_GATE_CONFIG)
    if gate_cfg:
        cfg.update(gate_cfg)

    if not cfg.get("enabled", True):
        return IngestDecision("store", "")

    # R1: compaction 摘要（lstrip 后前缀匹配，case-sensitive）
    if text.lstrip().startswith("[CONTEXT COMPACTION"):
        return IngestDecision("reject", "compaction")

    # R2: 超长
    max_len = int(cfg.get("max_len", 2000))
    if len(text) > max_len:
        return IngestDecision("reject", "oversize")

    # R3: 纯确认 / 状态问句 / 短指令
    stripped = _strip_text(text)
    low = stripped.lower()
    if low in _CONFIRM_BLACKLIST:
        return IngestDecision("reject", "low_signal_short")
    if _STATUS_QUESTION_RE.match(low):
        return IngestDecision("reject", "low_signal_short")
    if 0 < len(stripped) < 8:
        return IngestDecision("reject", "low_signal_short")

    # R4: 纯粘贴（无字母/汉字/数字）
    if _is_pure_paste(text):
        return IngestDecision("reject", "paste")

    # R5: builtin MEMORY.md 同步关闭
    if category == "memory_tool":
        return IngestDecision("reject", "builtin_dup")

    # R6: 同内容已存在 → 仅刷新状态（关键闭环，不覆盖原 content）
    if existing_meta and category in ("turn_memory", "conversation"):
        return IngestDecision("update_state", "state_only")

    # R7: 通过
    return IngestDecision("store", "")


# ---------------------------------------------------------------------------
# R6 执行函数：仅刷新 updated_at/count，绝不覆盖 content
# ---------------------------------------------------------------------------

def update_state_only(storage: Any, existing_meta: Dict[str, Any]) -> bool:
    """R6 命中后调用：仅刷新 updated_at/count，不覆盖 content。

    参数:
        storage: RedisStorage 实例（必须实现 _get_client()）。
        existing_meta: 至少含 "key" 字段（形如 "memory:frag:<hash>"）。

    返回:
        True = 刷新成功；False = 参数缺失或 storage 不可用或 Redis 异常。
    """
    from datetime import datetime, timezone

    if storage is None or not existing_meta:
        return False
    key = existing_meta.get("key")
    if not key:
        return False
    client = storage._get_client()  # noqa: SLF001 — 与 storage.store() 同级用法
    if not client:
        return False
    now = datetime.now(timezone.utc).isoformat()
    try:
        pipe = client.pipeline()
        pipe.hincrby(key, "touch_count", 1)
        pipe.hset(key, "updated_at", now)
        pipe.execute()
        return True
    except Exception:
        return False


__all__ = [
    "IngestDecision",
    "DEFAULT_GATE_CONFIG",
    "decide",
    "update_state_only",
]
