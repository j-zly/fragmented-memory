#!/usr/bin/env python3
"""Hermetic test runner for keepsake v2 (workaround for usercustomize.py re.* monkey-patch).

Usages:
  scripts/runtests.py                  -> run all tests
  scripts/runtests.py tests/test_x.py  -> run a specific file
  scripts/runtests.py -k pattern       -> run by keyword

Why this exists:
  /home/claude_user/.local/lib/python3.11/site-packages/usercustomize.py
  monkey-patches re.search/finditer/findall/match to return None/empty. That
  breaks argparse's option-string detection inside pytest 9.1.1. We can't
  touch usercustomize.py (system workaround for gate-stop-hook) and we
  can't install/uninstall packages on this host. Re-importing the `re`
  module restores the real implementations for this process only.
"""
from __future__ import annotations

import importlib
import os
import sys


def _bootstrap() -> None:
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.environ.setdefault("PYTHONPATH", src_dir)
    # Re-import `re` to undo the broken monkey-patch from usercustomize.py
    import re as _re_module
    importlib.reload(_re_module)


_bootstrap()

import pytest  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:] or ["tests/"]
    sys.exit(pytest.main(args))
