#!/usr/bin/env python3
"""
3D Medical Visualization — HTTP API Server 向后兼容导入桩
=============================================================

所有逻辑已迁移到 Visualization.API。此文件为重新导出桩（re-export shim），
确保直接 ``python visualization_server.py --port 8765`` 继续可用。

使用方式（推荐）：
    python Visualization/API.py --server --port 8765
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from API import server_main  # noqa: E402

if __name__ == "__main__":
    server_main()
