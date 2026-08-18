"""Gate A data preparation for v10.4 target-order planning.

Reads ONLY the `train` split of the frozen v10.2 splits file
(`results/clinical_window_v10_2/frozen/splits_v10_2.json`) and, with the fixed
seed 20260811, reorders the 512 scenes by scenario_id and divides them into

    planner_tune 384  -> tuning of pruning / beam width / look-ahead depth
    planner_gate 128  -> one-shot Gate A evaluation (planner frozen first)

Outputs:
    results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json
    results/clinical_window_v10_4_target_order/pilot_gate_a/SHA256SUMS

The v10.4 guide (Section 4.1) forbids reading any oracle_dev/probe/tuning/
validation/test/stress data for Gate A. This script touches only `train`.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

GATE_SEED = 20260811
PLANNER_TUNE_COUNT = 384
PLANNER_GATE_COUNT = 128

FROZEN_V102 = SIM / "results/clinical_window_v10_2/frozen/splits_v10_2.json"
OUT_DIR = SIM / "results/clinical_window_v10_4_target_order/pilot_gate_a"


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.loads(FROZEN_V102.read_text(encoding="utf-8"))
    if not payload.get("frozen"):
        raise RuntimeError("v10.2 splits must be frozen to seed Gate A")
    if payload.get("stage") != "d":
        raise RuntimeError(f"Gate A requires pure stage-d train, got stage={payload.get('stage')!r}")

    train = list(payload["splits"]["train"])
    if len(train) != 512:
        raise RuntimeError(f"expected exactly 512 train scenes, got {len(train)}")
    ids = [item["scenario_id"] for item in train]
    if len(set(ids)) != 512:
        raise RuntimeError("train scenario_ids are not unique")

    # Fixed seed, reorder by scenario_id first for reproducibility across runs.
    ordered = sorted(enumerate(train), key=lambda pair: pair[1]["scenario_id"])
    rng = random.Random(GATE_SEED)
    shuffled = list(ordered)
    rng.shuffle(shuffled)

    tune = [train[i] for i, _ in shuffled[:PLANNER_TUNE_COUNT]]
    gate = [train[i] for i, _ in shuffled[PLANNER_TUNE_COUNT:PLANNER_TUNE_COUNT + PLANNER_GATE_COUNT]]
    if len(tune) != PLANNER_TUNE_COUNT or len(gate) != PLANNER_GATE_COUNT:
        raise RuntimeError("unexpected split sizes after reorder")

    tune_ids = sorted(item["scenario_id"] for item in tune)
    gate_ids = sorted(item["scenario_id"] for item in gate)
    if set(tune_ids) & set(gate_ids):
        raise RuntimeError("planner_tune and planner_gate overlap")

    gate_a = {
        "version": "v10.4-gate-a-splits-v1",
        "seed": GATE_SEED,
        "source": str(FROZEN_V102.relative_to(SIM)),
        "source_frozen": True,
        "source_split": "train",
        "source_count": 512,
        "note": (
            "Gate A development data. planner_tune is for adjusting pruning, beam "
            "width and look-ahead depth; planner_gate is a one-shot Gate A "
            "evaluation after the planner is frozen. Derived ONLY from "
            "splits_v10_2.json train; no oracle_dev/probe/tuning/validation/test/"
            "stress data is read."
        ),
        "uses": {
            "planner_tune": "tune planner hyperparameters; inspect costs",
            "planner_gate": "one-shot Gate A GO/NO-GO evaluation",
        },
        "splits": {
            "planner_tune": {"count": len(tune), "scenario_ids": tune_ids, "scenarios": tune},
            "planner_gate": {"count": len(gate), "scenario_ids": gate_ids, "scenarios": gate},
        },
    }
    out_path = OUT_DIR / "gate_a_splits_v104.json"
    text = json.dumps(gate_a, ensure_ascii=False, indent=2) + "\n"
    out_path.write_text(text, encoding="utf-8")
    digest = _sha256_of_text(text)

    sha = (
        f"# Gate A splits (v10.4) hashes\n"
        f"# seed={GATE_SEED} source={FROZEN_V102.relative_to(SIM)}\n"
        f"{digest}  {out_path.name}\n"
        f"{_sha256_of_text(FROZEN_V102.read_text(encoding='utf-8'))}  splits_v10_2.json\n"
    )
    (OUT_DIR / "SHA256SUMS").write_text(sha, encoding="utf-8")

    print(f"planner_tune: {len(tune_ids)} scenes")
    print(f"planner_gate: {len(gate_ids)} scenes")
    print(f"overlap: {len(set(tune_ids) & set(gate_ids))}")
    print(f"wrote {out_path}")
    print(f"gate_a_splits_v104.json sha256: {digest}")


if __name__ == "__main__":
    main()
