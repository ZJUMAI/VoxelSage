#!/usr/bin/env python3
"""
Thread-safe progress tracker for long-running pipeline tasks.
"""

import threading
from typing import Any, Dict, Optional


class ProgressTracker:
    """
    Thread-safe progress reporter that writes progress into a shared dict.

    Usage:

        >>> jobs: dict = {}  # shared _jobs dict
        >>> jobs["my-job"] = {"status": "running", "progress": None}
        >>> tracker = ProgressTracker(target=jobs["my-job"])
        >>> tracker.update("Loading CT", 5)
        >>> tracker.update("VISTA3D segmentation", 10, "Initializing model...")
    """

    def __init__(self, target: Optional[Dict[str, Any]] = None):
        self._lock = threading.Lock()
        self._target = target

    def bind(self, target: Dict[str, Any]) -> None:
        """Bind or rebind to a target dict (e.g. _jobs[job_id])."""
        self._target = target

    def update(self, step: str, percentage: float, message: str = "") -> None:
        """
        Write progress into the target dict.

        Args:
            step: Short step name (e.g. "VISTA3D segmentation").
            percentage: 0-100 progress estimate.
            message: Optional detail message.
        """
        with self._lock:
            if self._target is not None:
                self._target["progress"] = {
                    "step": step,
                    "percentage": round(percentage, 1),
                    "message": message,
                }

    @property
    def current(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._target is None:
                return None
            return self._target.get("progress")


# Singleton convenience (for modules that import once, bind later)
_GLOBAL = ProgressTracker()


def get_global_tracker() -> ProgressTracker:
    return _GLOBAL
