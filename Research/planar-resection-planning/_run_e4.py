"""Spawn-safe launcher for E4 pilot."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["pilot_v108.py", "--scene-workers", "20", "--scenes", "32"]
runpy.run_path("pilot_v108.py", run_name="__main__")
