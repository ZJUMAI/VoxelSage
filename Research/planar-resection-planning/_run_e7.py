"""Spawn launcher for E7."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["benchmark_v108_lazy.py"]
runpy.run_path("benchmark_v108_lazy.py", run_name="__main__")
