"""Phase launcher."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import runpy
import sys
sys.argv = ["evaluate_v108_phase.py", "--controllers", sys.argv[1] if len(sys.argv) > 1 else "C4L", "--scene-workers", "20"]
runpy.run_path("evaluate_v108_phase.py", run_name="__main__")
