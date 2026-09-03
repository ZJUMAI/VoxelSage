"""Spawn launcher for E6."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["evaluate_v108_lazy.py", "--scene-workers", "20"]
runpy.run_path("evaluate_v108_lazy.py", run_name="__main__")
