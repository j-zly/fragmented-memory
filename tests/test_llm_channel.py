"""keepsake LLM 通道配置化 —— resolve_llm_channel 单元测试。

覆盖（任务书 G1/G2/G3/G4）:
  1. 有 llm 节 + key_file → resolve 出 base_url/model/api_key，
     URL 拼接正确断言 https://open.bigmodel.cn/api/paas/v4/chat/completions
  2. 无 llm 节 → resolve 回落 dashscope（现状回归，零配置=原样）
  3. key_file 不存在 → resolve 不抛，回落 _get_api_key 兜底链
  4. api_key 直填 优先于 key_file
  5. 日志无 key 泄漏：monkeypatch logger，断言任何 record 文本不含 key 值

设计要点:
  * 全 mock 无网络（任务书红线）
  * 测试用 tmp fake 文件 + 假 key 字符串（任务书红线）
  * 不依赖外部 hermes yaml/配置文件
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from keepsake.consolidator import (
    DASHSCOPE_BASE,
    DEFAULT_LLM_MODEL,
    resolve_llm_channel,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def fake_key_file(tmp_path):
    """写入一个 fake key 文件，返回路径。"""
    p = tmp_path / "fake_glm.pass"
    p.write_text("fake-key-do-not-use-9876\n")
    return str(p)


@pytest.fixture
def missing_key_file(tmp_path):
    """返回一个**不存在**的路径。"""
    return str(tmp_path / "does_not_exist.pass")


# ===========================================================================
# Test 1: llm 节 + key_file → resolve 出正确 base_url / model / api_key
# ===========================================================================

class TestResolveWithKeyFile:
    def test_bigmodel_channel_full_resolution(self, fake_key_file):
        """智谱免费端点 + key_file：全字段正确解析。"""
        cfg = {
            "llm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4-flash",
                "key_file": fake_key_file,
            }
        }
        ch = resolve_llm_channel(cfg)

        # base_url 不带尾 /
        assert ch["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert ch["model"] == "glm-4-flash"
        assert ch["api_key"] == "fake-key-do-not-use-9876"
        assert ch["source"] == "bigmodel"
        assert ch["key_file"] == fake_key_file

    def test_url_composition_bigmodel(self, fake_key_file):
        """G2: URL 拼接断言 https://open.bigmodel.cn/api/paas/v4/chat/completions
        （只到 /api/paas/v4，不加 /v1 —— 智谱路径比 OpenAI 少一层）。"""
        cfg = {
            "llm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4-flash",
                "key_file": fake_key_file,
            }
        }
        ch = resolve_llm_channel(cfg)
        url = f"{ch['base_url']}/chat/completions"
        assert url == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        # 关键负向断言：不能把 /v1 拼进 URL
        assert "/v1/chat" not in url

    def test_trailing_slash_normalized(self, fake_key_file):
        """base_url 带尾 / 也要兼容。"""
        cfg = {
            "llm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                "model": "glm-4-flash",
                "key_file": fake_key_file,
            }
        }
        ch = resolve_llm_channel(cfg)
        url = f"{ch['base_url']}/chat/completions"
        # rstrip 后不会重复 /
        assert "//chat" not in url
        assert url == "https://open.bigmodel.cn/api/paas/v4/chat/completions"


# ===========================================================================
# Test 2: 无 llm 节 → 回落 dashscope（零配置=原样）
# ===========================================================================

class TestResolveFallbackDashscope:
    def test_empty_cfg_falls_back_to_dashscope(self):
        """G3: 零配置=原样（向后兼容）。"""
        ch = resolve_llm_channel({})
        assert ch["base_url"] == DASHSCOPE_BASE
        assert ch["model"] == DEFAULT_LLM_MODEL
        assert ch["source"] == "dashscope_legacy"
        # api_key 走原 _get_api_key 链（测试环境无 env / 无 yaml → 期望空串）
        assert ch["api_key"] == ""

    def test_none_cfg_falls_back_to_dashscope(self):
        """cfg=None 也回落 dashscope。"""
        ch = resolve_llm_channel(None)
        assert ch["base_url"] == DASHSCOPE_BASE
        assert ch["model"] == DEFAULT_LLM_MODEL
        assert ch["source"] == "dashscope_legacy"

    def test_llm_empty_dict_falls_back(self):
        """cfg 存在但 llm 节是空 dict → 视为零配置回落。"""
        ch = resolve_llm_channel({"llm": {}})
        assert ch["base_url"] == DASHSCOPE_BASE
        assert ch["source"] == "dashscope_legacy"

    def test_url_composition_dashscope(self):
        """dashscope 路径现状回归（拼接 URL 仍正确）。"""
        ch = resolve_llm_channel({})
        url = f"{ch['base_url']}/chat/completions"
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


# ===========================================================================
# Test 3: key_file 不存在 → 不抛，回落 _get_api_key
# ===========================================================================

class TestResolveKeyFileMissing:
    def test_missing_key_file_does_not_raise(self, missing_key_file):
        """key_file 不存在 → resolve 必须不抛异常。"""
        cfg = {
            "llm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4-flash",
                "key_file": missing_key_file,
            }
        }
        # 不抛即过
        ch = resolve_llm_channel(cfg)
        # 端点 host 仍然按配置解析（base_url 不受 key_file 缺失影响）
        assert ch["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert ch["model"] == "glm-4-flash"
        # api_key 走兜底链（测试环境无 env 无 yaml → 期望空串，不抛）
        assert ch["api_key"] == ""
        # key_file 路径仍记录（便于诊断）
        assert ch["key_file"] == missing_key_file


# ===========================================================================
# Test 4: api_key 直填 优先于 key_file
# ===========================================================================

class TestResolveApiKeyPriority:
    def test_api_key_direct_beats_key_file(self, fake_key_file):
        """直填 api_key 优先于 key_file（即便 key_file 存在）。"""
        cfg = {
            "llm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4-flash",
                "key_file": fake_key_file,
                "api_key": "direct-inline-key-1234",
            }
        }
        ch = resolve_llm_channel(cfg)
        # 直填胜出
        assert ch["api_key"] == "direct-inline-key-1234"
        # 不应读取 key_file（用 sentinel 内容验证）
        assert "fake-key-do-not-use" not in ch["api_key"]

    def test_whitespace_only_key_file_stripped(self, fake_key_file):
        """key_file 内容含尾随空白 / 换行应被 strip。"""
        cfg = {
            "llm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4-flash",
                "key_file": fake_key_file,
            }
        }
        ch = resolve_llm_channel(cfg)
        # strip 已生效 —— 无尾随 \n / 空格
        assert ch["api_key"] == ch["api_key"].strip()
        assert "\n" not in ch["api_key"]
        assert " " not in ch["api_key"]


# ===========================================================================
# Test 5: 日志无 key 泄漏（G4）
# ===========================================================================

class _CaptureHandler(logging.Handler):
    """把所有 LogRecord 抓到列表里供断言。"""
    def __init__(self):
        super().__init__()
        self.records: list = []

    def emit(self, record):
        self.records.append(record)


class TestLogNoKeyLeak:
    def test_no_key_value_in_any_log_record(self, fake_key_file):
        """monkeypatch logger 后，断言任何 record 的文本不含 key 值。"""
        fake_key = "fake-key-do-not-use-9876"

        # consolidator logger 加 capture handler
        from keepsake import consolidator as cm
        cap = _CaptureHandler()
        cap.setLevel(logging.DEBUG)
        cm.logger.addHandler(cap)
        cm.logger.setLevel(logging.DEBUG)
        try:
            # 三种关键路径都跑一遍
            cfg_ok = {
                "llm": {
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "model": "glm-4-flash",
                    "key_file": fake_key_file,
                }
            }
            resolve_llm_channel(cfg_ok)

            # key_file 缺失路径（应该打 debug 而非 error）
            cfg_missing = {
                "llm": {
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "model": "glm-4-flash",
                    "key_file": "/no/such/file/12345",
                }
            }
            resolve_llm_channel(cfg_missing)

            # 兜底链路径（无 llm 节）
            resolve_llm_channel({})
        finally:
            cm.logger.removeHandler(cap)

        # 关键断言：所有 LogRecord 的格式化文本都不含 key 内容
        leaked = []
        for rec in cap.records:
            try:
                msg = rec.getMessage()
            except Exception:
                msg = str(rec)
            if fake_key in msg:
                leaked.append((rec.levelname, msg))

        assert not leaked, (
            f"key value leaked into log records: {leaked}"
        )

    def test_no_key_in_debug_message_format(self, fake_key_file, caplog):
        """额外 caplog 验证：debug 日志只打印路径和长度，不打 key。"""
        cfg = {
            "llm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4-flash",
                "key_file": fake_key_file,
            }
        }
        with caplog.at_level(logging.DEBUG, logger="keepsake.consolidator"):
            resolve_llm_channel(cfg)

        all_text = "\n".join(rec.getMessage() for rec in caplog.records)
        # 关键负向断言
        assert "fake-key-do-not-use-9876" not in all_text
        # 正向断言：路径出现
        assert fake_key_file in all_text
