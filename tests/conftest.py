"""测试 bootstrap：仅在脱离 Hermes 的独立环境生效。

正式运行环境（pip install keepsake 到 Hermes 镜像里）会从 `agent` /
`tools` 包拿到真实实现；本文件的存在只为让 `pytest` 在 CI / agent 工位
上能直跑 `import keepsake`。
"""

from __future__ import annotations

import sys
import types


def _install_stubs() -> None:
    if "agent" in sys.modules and "tools" in sys.modules:
        return

    # agent.memory_provider.MemoryProvider —— KeepsakeProvider 仅 super().__init__()
    agent_mod = types.ModuleType("agent")
    agent_mp_mod = types.ModuleType("agent.memory_provider")

    class _StubMemoryProvider:
        def __init__(self, *args, **kwargs):
            pass

    agent_mp_mod.MemoryProvider = _StubMemoryProvider
    agent_mod.memory_provider = agent_mp_mod
    sys.modules["agent"] = agent_mod
    sys.modules["agent.memory_provider"] = agent_mp_mod

    # tools.registry.tool_error —— 调用点只把它当字符串返回用
    tools_mod = types.ModuleType("tools")
    tools_reg_mod = types.ModuleType("tools.registry")

    def _stub_tool_error(msg):
        return f"[tool_error] {msg}"

    tools_reg_mod.tool_error = _stub_tool_error
    tools_mod.registry = tools_reg_mod
    sys.modules["tools"] = tools_mod
    sys.modules["tools.registry"] = tools_reg_mod


_install_stubs()
