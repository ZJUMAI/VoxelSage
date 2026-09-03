"""E8 sensitivity aggregation: per-controller, per-condition summary.

Overrun definition (corrected per Bryce 2026-09-04): an episode is
considered overrun when its realized total blood loss
``realized_episode_B_ml`` exceeds the per-episode budget
``budget_ml = baseline_blood + margin``.  Earlier versions incorrectly
used ``selected_max_B_total_ml`` (max B in any single planned step),
which missed the accumulated tail B and falsely reported zero overruns
on C4L S1/S2.

The fallback rewrite (commit c8ea287+1) introduced an ``infeasible``
terminal state: when every safe candidate is rejected, the controller
terminates with ``failure_reason == "infeasible_no_safe_candidate"``
and ``completion == false``.  These episodes do NOT count as overruns
(they do not commit to an unsafe plan), but they are reported
separately as ``infeasibles``.

Shards captured before the fallback rewrite (e.g. C4L on S1, S2) may
still show the older "fall back to serpentine S" behaviour; for those
the ``failure_reason`` will be ``None`` and ``realized_episode_B_ml``
reflects the unsafe S path.  Re-running those shards is the v10.8
follow-up plan (see also E8 128-scene补跑).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
SENS = V108 / "sensitivity"
REPORT = V108 / "sensitivity_summary.json"

INFEASIBLE_FAILURE_REASONS = {"infeasible_no_safe_candidate", "all_candidates_unsafe"}


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
    fallback_version_note = (
        "S1/S2 C4L shards captured before the infeasible fallback rewrite; "
        "realized_episode_B_ml reflects the unsafe S-fallback path."
    )
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
            infeasibles = 0
            realized_B_values = []
            budget_values = []
            for f in ctrl.glob("*.json"):
                try:
                    j = json.loads(f.read_text())
                except Exception:
                    continue
                walls.append(float(j.get("wall_seconds", 0.0)))
                if j.get("completion", False):
                    completes += 1
                if int(j.get("safety_invariant_violations", 0)) > 0:
                    invariants += 1
                realized = float(j.get("realized_episode_B_ml", 0.0))
                budget = float(j.get("budget_ml", 0.0))
                failure = j.get("failure_reason") or ""
                is_infeasible = (
                    failure in INFEASIBLE_FAILURE_REASONS
                    or bool(j.get("infeasible", False))
                    or int(j.get("infeasible_count", 0)) > 0
                )
                if is_infeasible:
                    infeasibles += 1
                elif realized > budget + 1e-9:
                    overruns += 1
                realized_B_values.append(realized)
                budget_values.append(budget)
            per_ctrl[ctrl.name] = {
                "wall": _stats(walls),
                "completes": completes,
                "invariants": invariants,
                "overruns": overruns,
                "infeasibles": infeasibles,
                "n_shards": len(realized_B_values),
            }
        out["per_controller_per_condition"][cond.name] = per_ctrl
    out["overrun_definition"] = (
        "realized_episode_B_ml > budget_ml (per-episode baseline+margin); "
        "infeasible episodes are reported separately and not counted as overruns"
    )
    out["fallback_semantics"] = (
        "all-unsafe -> terminate with failure_reason=infeasible_no_safe_candidate; "
        "no unsafe S action is executed"
    )
    out["notes"] = fallback_version_note
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[E8] wrote {REPORT}")
    print()
    print("Per-condition C4L (overrun / infeasible / completes):")
    for cond in out["conditions"]:
        c4l = out["per_controller_per_condition"][cond].get("C4L", {})
        if c4l:
            n = c4l.get("n_shards", 0)
            print(f"  {cond}: n={n:3d}  completes={c4l.get('completes', 0):3d}  "
                  f"overruns={c4l.get('overruns', 0):3d}  infeasibles={c4l.get('infeasibles', 0):3d}")
    print()
    print("C5 sensitivity check (should have low overrun if any):")
    for cond in out["conditions"]:
        c5 = out["per_controller_per_condition"][cond].get("C5", {})
        if c5:
            print(f"  {cond}: n={c5.get('n_shards', 0):3d}  completes={c5.get('completes', 0):3d}  "
                  f"invariants={c5.get('invariants', 0):3d}  overruns={c5.get('overruns', 0):3d}")


if __name__ == "__main__":
    main()
