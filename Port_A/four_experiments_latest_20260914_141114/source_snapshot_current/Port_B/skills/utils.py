"""
Skill 通用工具函数
===================
被内置 Skills 和用户上传 Skills 共享使用。
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np


def convert_numpy(obj: Any) -> Any:
    """递归将 numpy 类型转为 Python 原生类型（JSON 安全）。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj
