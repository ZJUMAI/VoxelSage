"""Spawn-safe launcher: re-exec equivalence_v108.py as __main__."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Replace this process with equivalence_v108.py as __main__ so that
# multiprocessing.spawn can re-execute it cleanly in worker children.
import runpy
sys.argv = ["equivalence_v108.py", "--scene-workers", "20"]
runpy.run_path("equivalence_v108.py", run_name="__main__")
