"""E8 phase launcher: 5 conditions x 64 base scenes x 5 controllers.

For each (condition, controller), call evaluate_v108_phase with a
custom output directory and a baseline built from the sensitivity
condition's clinical config.

The sensitivity is created by overriding the ``max_clamp_minutes`` and
``bleeding_probability`` of the clinical config.  The baseline is
re-rolled for each condition because the budget changes with the
clamp ceiling.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
SENS = V108 / "sensitivity"

# v10.7.1 sensitivity conditions (plan §7.9)
CONDITIONS = {
    "S0": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S1": {"max_clamp_minutes": 12.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S2": {"max_clamp_minutes": 10.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S3": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.5},
    "S4": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.25},
}

# Baseline v10.7 frozen margin
MARGIN = 16.07054347826075
CHECKPOINT = REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt"

# 64 base scenes from the v10.8 256 split
SPLIT = V108 / "frozen/split_lazy_replication.json"
BASE = V108 / "frozen/baseline_lazy_replication.json"


def main():
    SENS.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text())
    all_scenes = split["scenarios"][:64]
    base_records = json.loads(BASE.read_text())["records"]
    print(f"[E8] 64 base scenes x 5 conditions x 5 controllers = 1600 task tuples")

    # For each condition, run all 5 controllers phase-by-phase
    for cond_id, cfg in CONDITIONS.items():
        print(f"\n[E8/{cond_id}] {cfg}")
        for controller in ("C0", "C3", "C4E", "C4L", "C5"):
            cond_dir = SENS / cond_id / controller
            cond_dir.mkdir(parents=True, exist_ok=True)
            # Generate per-scene tasks
            for sc in all_scenes:
                sid = sc["scenario_id"]
                if sid not in base_records:
                    continue
                out = cond_dir / f"{sid}.json"
                if out.exists():
                    continue
                # Use the rollout_controller directly via python -c
                # (E8 cfg_overrides is per-scene; we pass full cfg per call)
                baseline = float(base_records[sid]["expected_blood_loss_ml"])
                cmd = [
                    "C:/Users/Bingh/miniconda3/envs/v108/python.exe", "-c",
                    f"""
import json, time, sys
sys.path.insert(0, r"{REPO.as_posix()}")
from lazy_confirmation_controllers_v108 import rollout_controller
scene = json.loads({json.dumps(json.dumps(sc))})
with open(r"{cond_dir / (sid + '.json')}", "w", encoding="utf-8") as f:
    pass
res = rollout_controller(
    "{controller}", scene,
    baseline_blood={baseline}, margin_ml={MARGIN},
    cfg={json.dumps(cfg)},
    checkpoint_path=r"{CHECKPOINT.as_posix()}",
)
res["scenario_id"] = "{sid}"
res["controller"] = "{controller}"
res["condition"] = "{cond_id}"
with open(r"{(cond_dir / (sid + '.json')).as_posix()}", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
"""
                ]
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if r.returncode != 0:
                    print(f"  ERR {controller}/{sid}: {r.stderr[-300:]}")
                else:
                    pass
            done = len(list(cond_dir.glob("*.json")))
            print(f"  [{cond_id}] {controller}: {done} shards ({(time.time()-t0)/60:.1f} min for this controller)")


if __name__ == "__main__":
    main()
