"""Spawn launcher for E5 (and any other module that wants __main__)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["prepare_v108_lazy_split.py", "--workers", "20"]
runpy.run_path("prepare_v108_lazy_split.py", run_name="__main__")
