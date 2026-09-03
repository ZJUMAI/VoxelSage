"""E8 sensitivity aggregation: per-controller, per-condition summary."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
SENS = V108 / "sensitivity"
REPORT = V108 / "sensitivity_summary.json"


def _stats(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": s[n // 2],
        "p95": s[int(0.95 * (n - 1))] if n > 1 else s[0],
        "max": s[-1],
    }


def main():
    out: dict = {"conditions": [], "per_controller_per_condition": {}}
    for cond in sorted(SENS.iterdir()):
        if not cond.is_dir():
            continue
        out["conditions"].append(cond.name)
        per_ctrl: dict = {}
        for ctrl in sorted(cond.iterdir()):
            if not ctrl.is_dir():
                continue
            walls = []
            completes = 0
            invariants = 0
            overruns = 0
            for f in ctrl.glob("*.json"):
                try:
                    j = json.loads(f.read_text())
                    walls.append(float(j.get("wall_seconds", 0.0)))
                    if j.get("completion", False):
                        completes += 1
                    if int(j.get("safety_invariant_violations", 0)) > 0:
                        invariants += 1
                    if float(j.get("selected_max_B_total_ml", 0)) > float(j.get("budget_ml", 0)) + 1e-9:
                        overruns += 1
                except Exception:
                    pass
            per_ctrl[ctrl.name] = {
                "wall": _stats(walls),
                "completes": completes,
                "invariants": invariants,
                "overruns": overruns,
            }
        out["per_controller_per_condition"][cond.name] = per_ctrl
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[E8] wrote {REPORT}")
    print()
    print("Per-condition C4L wall time (sensitivity):")
    for cond in out["conditions"]:
        stats = out["per_controller_per_condition"][cond].get("C4L", {}).get("wall", {})
        if stats:
            print(f"  {cond}: n={stats['n']:3d}  p50={stats['median']:.2f}  p95={stats['p95']:.2f}  max={stats['max']:.2f}")
    print()
    print("C5 sensitivity check (should have low overrun if any):")
    for cond in out["conditions"]:
        c5 = out["per_controller_per_condition"][cond].get("C5", {})
        if c5:
            print(f"  {cond}: n={c5.get('completes', '?')}/64 completes  invariants={c5.get('invariants', '?')}  overruns={c5.get('overruns', '?')}")


if __name__ == "__main__":
    main()
