"""
多 GPU 管理器
==============
自动检测空闲 GPU 并分配，支持并发安全。

用法:
    from Tool_Box.gpu_manager import GPUManager

    manager = GPUManager()
    with manager.allocate() as device:
        # 在此块中运行 GPU 任务
        print(f"Using {device}")
"""

import os
import re
import subprocess
import time
import logging
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# 最小可用显存（显存低于此值的 GPU 视为不可用）
_MIN_FREE_MEMORY_MB = 2048  # 2 GB

# nvidia-smi 查询超时（秒）— 防止 GPU 驱动卡死导致事件循环阻塞
_NV_SMI_TIMEOUT = 5


def query_gpu_info() -> List[Dict[str, object]]:
    """查询所有 GPU 的型号和显存信息。

    Returns:
        [{index, name, memory_total_mb, memory_used_mb, memory_free_mb}, ...]
        查询失败返回空列表。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_NV_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []

    gpus = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(", ")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "memory_free_mb": int(parts[4]),
            })
        except (ValueError, IndexError):
            continue
    return gpus


def print_gpu_info() -> None:
    """打印所有 GPU 状态到日志。"""
    gpus = query_gpu_info()
    if not gpus:
        logger.warning("No GPU detected via nvidia-smi")
        return
    for g in gpus:
        pct = (g["memory_used_mb"] / max(g["memory_total_mb"], 1)) * 100
        logger.info(
            f"  GPU {g['index']}: {g['name']}  "
            f"{g['memory_used_mb']}/{g['memory_total_mb']} MB "
            f"({pct:.0f}% used)"
        )


def _find_best_gpu(
    gpu_infos: List[Dict[str, object]],
    exclude_indices: Set[int],
) -> Optional[int]:
    """从 GPU 信息中选剩余显存最大的 GPU。

    Args:
        gpu_infos: query_gpu_info() 返回的列表。
        exclude_indices: 排除的 GPU 索引集合（已在用）。

    Returns:
        GPU index，或 None（无可用 GPU）。
    """
    candidates = [
        g for g in gpu_infos
        if g["index"] not in exclude_indices
        and g["memory_free_mb"] >= _MIN_FREE_MEMORY_MB
    ]
    if not candidates:
        return None
    # 选剩余显存最大的
    candidates.sort(key=lambda g: g["memory_free_mb"], reverse=True)
    return candidates[0]["index"]


class GPUManager:
    """多 GPU 管理器。

    负责自动选择空闲 GPU、标记占用 / 释放、防止同一 GPU 被并发抢占。
    线程安全。

    Usage:
        gpu_mgr = GPUManager()

        # 自动选卡
        with gpu_mgr.allocate() as dev:
            run_segmentation(device=dev)

        # 手动指定设备
        with gpu_mgr.allocate(preferred="cuda:2") as dev:
            run_segmentation(device=dev)
    """

    def __init__(self, min_free_memory_mb: int = _MIN_FREE_MEMORY_MB):
        self._lock = threading.Lock()
        self._in_use: Set[int] = set()  # 当前被此进程占用的 GPU index
        self._min_free = min_free_memory_mb

    # ── public API ─────────────────────────────────────────────

    def get_available_gpu(self, preferred: Optional[str] = None) -> str:
        """获取一个可用的 CUDA 设备字符串。

        Args:
            preferred: 用户手动指定的设备（如 "cuda:2"）。
                       若提供则直接用（不作空闲检查，但标记占用）。

        Returns:
            "cuda:<N>" 字符串。
        """
        if preferred is not None:
            idx = self._parse_device(preferred)
            if idx is not None:
                with self._lock:
                    self._in_use.add(idx)
                return preferred

        # 自动选择：nvidia-smi 查询和 _in_use 标记在同一锁内，防止竞态
        with self._lock:
            gpus = query_gpu_info()
            if not gpus:
                # nvidia-smi 不可用，回退 cuda:0
                logger.warning("nvidia-smi query failed, falling back to cuda:0")
                self._in_use.add(0)
                return "cuda:0"

            best = _find_best_gpu(gpus, self._in_use)
            if best is None:
                # 所有 GPU 都不够空闲——选剩余最多的（可能 OOM，但尽力了）
                available = [g for g in gpus if g["index"] not in self._in_use]
                if available:
                    available.sort(key=lambda g: g["memory_free_mb"], reverse=True)
                    best = available[0]["index"]
                    logger.warning(
                        f"No GPU with >= {self._min_free}MB free. "
                        f"Best effort: GPU {best} ({available[0]['memory_free_mb']}MB free)"
                    )
                else:
                    # 所有 GPU 都被标记在用——等待重试由外层调用方处理
                    raise RuntimeError(
                        f"All GPUs are currently in use ({len(self._in_use)}/{len(gpus)}). "
                        "Retry later."
                    )

            self._in_use.add(best)
            device = f"cuda:{best}"
            logger.info(
                f"Allocated GPU {best} "
                f"(free: {next((g['memory_free_mb'] for g in gpus if g['index'] == best), '?')}MB)"
            )
            return device

    def release_gpu(self, device: str) -> None:
        """释放指定 GPU，允许后续请求使用。"""
        idx = self._parse_device(device)
        if idx is None:
            return
        with self._lock:
            self._in_use.discard(idx)
        logger.info(f"Released GPU {idx}")

    @contextmanager
    def allocate(self, preferred: Optional[str] = None):
        """上下文管理器：进入时获取 GPU，退出时释放。

        Args:
            preferred: 首选设备（如 "cuda:1"），或 None 自动选择。

        Yields:
            "cuda:<N>" 设备字符串。
        """
        device = self.get_available_gpu(preferred)
        try:
            yield device
        finally:
            self.release_gpu(device)

    def all_free(self) -> bool:
        """是否所有 GPU 都未被占用。"""
        with self._lock:
            return len(self._in_use) == 0

    def in_use_count(self) -> int:
        """当前占用中的 GPU 数量。"""
        with self._lock:
            return len(self._in_use)

    def summary(self) -> dict:
        """返回 GPU 状态摘要。"""
        gpus = query_gpu_info()
        with self._lock:
            in_use = list(self._in_use)
        devices = [f"cuda:{i}" for i in sorted(in_use)]
        return {
            "detected": len(gpus),
            "in_use": in_use,
            "devices": devices,
            "gpus": gpus,
        }

    # ── internal ────────────────────────────────────────────

    @staticmethod
    def _parse_device(device: str) -> Optional[int]:
        """从 "cuda:N" 提取 N。失败返回 None。"""
        if not device or not isinstance(device, str):
            return None
        m = re.match(r"^cuda:(\d+)$", device.strip())
        if m:
            return int(m.group(1))
        return None
