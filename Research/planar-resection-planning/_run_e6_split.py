"""Spawn launcher for E6 split (fast + slow passes)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
# Run slow first since fast already partially done
sys.argv = ["evaluate_v108_lazy_split.py", "--scene-workers", "20", "--passes", "slow,fast"]
runpy.run_path("evaluate_v108_lazy_split.py", run_name="__main__")
