"""Spawn launcher for E8."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["sensitivity_v108_lazy.py", "--scene-workers", "20", "--limit", "128"]
runpy.run_path("sensitivity_v108_lazy.py", run_name="__main__")
