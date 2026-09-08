"""写闸门（ingest_gate v1，2026-09）单元测试。

纯单测，不依赖 Redis / Hermes runtime。

每个 R1-R7 至少 2 个用例（正反），命名规则 test_<规则>_<场景>_<期望>，
便于 GATES G4 直接 grep 对应关系。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from keepsake.ingest_gate import (
    DEFAULT_GATE_CONFIG,
    IngestDecision,
    decide,
    update_state_only,
)


# ===========================================================================
# R1 — compaction 摘要
# ===========================================================================

class TestR1Compaction:
    """R1：text.lstrip().startswith("[CONTEXT COMPACTION") → reject / compaction"""

    def test_r1_compaction_prefix_rejects(self):
        d = decide("[CONTEXT COMPACTION] 这是一段压缩摘要，含 30 条过期任务。", "turn_memory")
        assert d == IngestDecision("reject", "compaction")

    def test_r1_normal_text_stores(self):
        d = decide("请帮我把前端分页组件的样式调整一下", "turn_memory")
        assert d.action == "store"
        assert d.reason == ""

    def test_r1_case_sensitive_lowercase_does_not_match(self):
        """R1 大小写敏感 —— 小写前缀不走 R1，会被 R3 长度阈值兜底。"""
        d = decide("[ context compaction] blah blah blah 长一点确保超过 8 字符", "turn_memory")
        assert d.action == "store"

    def test_r1_leading_whitespace_still_matches(self):
        """R1 用 lstrip 处理前导空白后再判前缀。"""
        d = decide("   \n[CONTEXT COMPACTION] 摘要正文", "turn_memory")
        assert d == IngestDecision("reject", "compaction")


# ===========================================================================
# R2 — 超长
# ===========================================================================

class TestR2Oversize:
    """R2：len(text) > max_len → reject / oversize（max_len 默认 2000）"""

    def test_r2_over_2000_rejects(self):
        d = decide("A" * 2001, "turn_memory")
        assert d == IngestDecision("reject", "oversize")

    def test_r2_exactly_2000_stores(self):
        """R2 是「严格 >」，2000 不算超长。"""
        d = decide("A" * 2000, "turn_memory")
        assert d.action == "store"

    def test_r2_max_len_override_respected(self):
        d = decide("A" * 101, "turn_memory", gate_cfg={"max_len": 100})
        assert d == IngestDecision("reject", "oversize")

    def test_r2_max_len_override_relaxes(self):
        d = decide("A" * 5000, "turn_memory", gate_cfg={"max_len": 10000})
        assert d.action == "store"


# ===========================================================================
# R3 — 纯确认 / 状态问句 / 短指令
# ===========================================================================

class TestR3LowSignalShort:
    """R3：剥离标点 + emoji 后命中黑名单 / 状态问句正则 / len(stripped) < 8 → reject / low_signal_short"""

    # -- 黑名单 --

    def test_r3_blacklist_ke_deyi_rejects_with_punct(self):
        d = decide("可以的。", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_blacklist_ke_deyi_rejects_with_emoji(self):
        d = decide("👍 可以的", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_blacklist_ke_deyi_with_clause_stores(self):
        """任务书示例：「可以，但是把 CH 内存闸改成 0.5」→ store。"""
        d = decide("可以，但是把 CH 内存闸改成 0.5", "turn_memory")
        assert d.action == "store"

    def test_r3_blacklist_bushu_single_char_rejects(self):
        d = decide("部署", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_blacklist_bushu_with_clause_stores(self):
        d = decide("部署方案选 A 先灰度", "turn_memory")
        assert d.action == "store"

    def test_r3_blacklist_ok_lowercase(self):
        d = decide("ok", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_blacklist_ok_uppercase(self):
        d = decide("OK", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_blacklist_continue_with_clause_stores(self):
        """黑名单外的「继续XXX」走长度判定。"""
        d = decide("继续把完整方案执行到底", "turn_memory")
        assert d.action == "store"

    # -- 状态问句 --

    def test_r3_status_question_prefix_rejects(self):
        d = decide("目前情况", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_status_question_with_full_width_question_mark_rejects(self):
        d = decide("目前情况？", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_status_question_with_half_width_question_mark_rejects(self):
        d = decide("弄好了吗?", "turn_memory")
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_non_status_long_question_stores(self):
        """非状态问句前缀的长问句应正常 store。"""
        d = decide("为什么这次接口超时这么严重呢", "turn_memory")
        assert d.action == "store"

    # -- 短指令（剥离后 < 8 字符） --

    def test_r3_short_non_blacklist_rejects(self):
        """剥离后 len < 8 → reject。"""
        d = decide("重启吧", "turn_memory")  # strip 后「重启吧」= 3 字符
        assert d == IngestDecision("reject", "low_signal_short")

    def test_r3_long_non_blacklist_stores(self):
        d = decide("重启完整方案并跑回归", "turn_memory")
        assert d.action == "store"


# ===========================================================================
# R4 — 纯粘贴
# ===========================================================================

class TestR4Paste:
    """R4：text 中无任何字母/汉字/数字 → reject / paste"""

    def test_r4_dashes_rejects(self):
        d = decide("---", "turn_memory")
        assert d == IngestDecision("reject", "paste")

    def test_r4_equals_rejects(self):
        d = decide("=================================================", "turn_memory")
        assert d == IngestDecision("reject", "paste")

    def test_r4_markdown_table_separator_rejects(self):
        d = decide("|---|---|---|", "turn_memory")
        assert d == IngestDecision("reject", "paste")

    def test_r4_url_with_letters_stores(self):
        """URL 含字母（https），不会命中 R4。"""
        d = decide("请打开 https://example.com 看一下文档", "turn_memory")
        assert d.action == "store"

    def test_r4_text_with_chinese_stores(self):
        d = decide("--- 今天的工作总结以及明天的安排 ---", "turn_memory")
        assert d.action == "store"


# ===========================================================================
# R5 — builtin MEMORY.md 同步关闭
# ===========================================================================

class TestR5BuiltinDup:
    """R5：category == "memory_tool" → reject / builtin_dup"""

    def test_r5_memory_tool_rejects(self):
        d = decide("这是一条 builtin MEMORY.md 条目", "memory_tool")
        assert d == IngestDecision("reject", "builtin_dup")

    def test_r5_other_category_stores(self):
        d = decide("正常的 turn_memory 内容，长度足够", "turn_memory")
        assert d.action == "store"

    def test_r5_r6_shortcut_order_check(self):
        """R5 在 R6 之前 —— 即便 existing_meta 存在，memory_tool 仍 reject。"""
        d = decide(
            "这是一条 builtin MEMORY.md 条目，长度足够",
            "memory_tool",
            existing_meta={"key": "memory:frag:abc"},
        )
        assert d == IngestDecision("reject", "builtin_dup")


# ===========================================================================
# R6 — 同内容已存在 → 仅刷新状态
# ===========================================================================

class TestR6UpdateState:
    """R6：existing_meta 存在 + category in (turn_memory, conversation) → update_state / state_only"""

    def test_r6_turn_memory_with_existing_meta_updates_state(self):
        meta = {"key": "memory:frag:abc123def456"}
        d = decide("今天处理了一批订单结果如何呢", "turn_memory", existing_meta=meta)
        assert d == IngestDecision("update_state", "state_only")

    def test_r6_conversation_with_existing_meta_updates_state(self):
        meta = {"key": "memory:frag:abc123def456"}
        d = decide("今天处理了一批订单结果如何呢", "conversation", existing_meta=meta)
        assert d == IngestDecision("update_state", "state_only")

    def test_r6_no_existing_meta_stores(self):
        d = decide("今天处理了一批订单结果如何呢", "turn_memory", existing_meta=None)
        assert d.action == "store"

    def test_r6_empty_existing_meta_stores(self):
        """空 dict 视同 falsy —— 走 R7 store。"""
        d = decide("今天处理了一批订单结果如何呢", "turn_memory", existing_meta={})
        assert d.action == "store"

    def test_r6_other_category_with_existing_meta_stores(self):
        """不在 (turn_memory, conversation) 的 category + 已有 meta → 不触发 R6，走 store。"""
        meta = {"key": "memory:frag:abc"}
        d = decide("这是一条事实条目记录用的", "fact", existing_meta=meta)
        assert d.action == "store"

    def test_r6_shortcut_after_r5_memory_tool(self):
        """R5 优先于 R6 —— category='memory_tool' 即使有 existing_meta 也走 R5 reject。"""
        meta = {"key": "memory:frag:abc"}
        d = decide(
            "这是一条 MEMORY.md 内容，长度足够",
            "memory_tool",
            existing_meta=meta,
        )
        assert d == IngestDecision("reject", "builtin_dup")


# ===========================================================================
# R7 — 通过
# ===========================================================================

class TestR7Store:
    """R7：未命中 R1-R6 → store"""

    def test_r7_normal_chinese_text_stores(self):
        d = decide("明天的会议改到下午三点开始请准时参加", "turn_memory")
        assert d.action == "store"

    def test_r7_normal_english_text_stores(self):
        d = decide("please update the README with new section today", "turn_memory")
        assert d.action == "store"

    def test_r7_reason_is_empty(self):
        d = decide("正常内容长度足够不会触发任何规则", "turn_memory")
        assert d.reason == ""


# ===========================================================================
# 顺序短路（cross-cutting）
# ===========================================================================

class TestShortCircuitOrdering:
    """R1-R7 顺序短路：每条规则一旦命中立刻返回，不被后续规则覆盖。"""

    def test_order_r1_before_r2(self):
        """R1 (compaction) 优先级 > R2 (oversize) —— 超长 compaction 也按 R1 reject。"""
        text = "[CONTEXT COMPACTION] " + ("A" * 5000)
        d = decide(text, "turn_memory")
        assert d.reason == "compaction"

    def test_order_r2_before_r3(self):
        """R2 优先级 > R3 —— 超长黑名单词也按 R2 reject。"""
        d = decide("可以" + "A" * 2100, "turn_memory")
        assert d.reason == "oversize"

    def test_order_r3_before_r4(self):
        """R3 优先级 > R4 —— 短确认词即使「看上去是纯粘贴」也按 R3 reject。"""
        d = decide("可以", "turn_memory")
        assert d.reason == "low_signal_short"

    def test_order_r4_before_r5(self):
        """R4 优先级 > R5 —— 纯粘贴即使 category=memory_tool 也按 R4 reject。"""
        d = decide("---", "memory_tool")
        assert d.reason == "paste"

    def test_order_r5_before_r6(self):
        """R5 优先级 > R6 —— memory_tool + existing_meta → R5 reject。"""
        d = decide("这是一条 builtin MEMORY.md 内容", "memory_tool", existing_meta={"key": "memory:frag:abc"})
        assert d.reason == "builtin_dup"

    def test_order_r6_before_r7(self):
        """R6 优先级 > R7 —— existing_meta + turn_memory → R6 update_state。"""
        d = decide("正常内容足够长以通过 R3 阈值", "turn_memory", existing_meta={"key": "memory:frag:abc"})
        assert d.action == "update_state"


# ===========================================================================
# 配置开关
# ===========================================================================

class TestGateConfig:
    """DEFAULT_GATE_CONFIG 默认值；enabled=false 时全部短路到 store。"""

    def test_default_config_enabled_and_max_len(self):
        assert DEFAULT_GATE_CONFIG == {"enabled": True, "max_len": 2000}

    def test_disabled_short_circuits_all_to_store(self):
        gate = {"enabled": False}
        for text in (
            "[CONTEXT COMPACTION] x",
            "A" * 5000,
            "可以的。",
            "---",
            "any",
        ):
            d = decide(text, "turn_memory", gate_cfg=gate)
            assert d.action == "store", f"text={text!r} should store when disabled, got {d}"

    def test_custom_max_len_respected(self):
        gate = {"max_len": 10}
        assert decide("A" * 11, "turn_memory", gate_cfg=gate).reason == "oversize"
        assert decide("A" * 10, "turn_memory", gate_cfg=gate).action == "store"


# ===========================================================================
# update_state_only 行为（不连 Redis，靠 fake 验证 client 调用面）
# ===========================================================================

class _FakePipeline:
    def __init__(self, calls):
        self.calls = calls

    def hincrby(self, key, field, n):
        self.calls.append(("hincrby", key, field, n))

    def hset(self, key, field, value):
        self.calls.append(("hset", key, field, value))

    def execute(self):
        return []


class _FakeClient:
    def __init__(self):
        self.calls = []  # list of ("hincrby"|"hset", key, field, value)

    def pipeline(self):
        return _FakePipeline(self.calls)


class _FakeStorage:
    def __init__(self):
        self.client = _FakeClient()

    def _get_client(self):
        return self.client


class TestUpdateStateOnlyHelper:
    """update_state_only 是 R6 的执行函数，验证其对 storage 的调用面。
    不连 Redis —— 用 fake 记录 hset/hincrby 调用次数与键。
    """

    def test_update_state_bumps_touch_count_and_updated_at(self):
        st = _FakeStorage()
        meta = {"key": "memory:frag:abc"}
        ok = update_state_only(st, meta)
        assert ok is True
        # 必有 hincrby(touch_count, 1) 和 hset(updated_at, <iso>)
        ops = st.client.calls
        assert ("hincrby", "memory:frag:abc", "touch_count", 1) in ops
        hset_updated = [c for c in ops if c[0] == "hset" and c[2] == "updated_at"]
        assert len(hset_updated) == 1
        # ISO 格式验证
        from datetime import datetime
        datetime.fromisoformat(hset_updated[0][3])

    def test_update_state_returns_false_when_no_meta(self):
        st = _FakeStorage()
        assert update_state_only(st, None) is False
        assert update_state_only(st, {}) is False
        assert st.client.calls == []

    def test_update_state_returns_false_when_no_client(self):
        class _Empty:
            def _get_client(self):
                return None

        assert update_state_only(_Empty(), {"key": "memory:frag:abc"}) is False
