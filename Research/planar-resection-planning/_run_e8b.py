"""Spawn launcher for E8 sensitivity (in-process phase)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["evaluate_v108_sensitivity.py", "--scene-workers", "20", "--limit", "64"]
runpy.run_path("evaluate_v108_sensitivity.py", run_name="__main__")
