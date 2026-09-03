"""Spawn launcher for E6 inline (in-worker shard write)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["evaluate_v108_lazy_inline.py", "--scene-workers", "20", "--passes", "slow,fast"]
runpy.run_path("evaluate_v108_lazy_inline.py", run_name="__main__")
