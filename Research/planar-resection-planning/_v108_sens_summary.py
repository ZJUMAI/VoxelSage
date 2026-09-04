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

Legacy S1/S2 directories may still contain the older "fall back to
serpentine S" behaviour.  Pass a fresh isolated result directory with
``--final-c4l-root`` to replace those cells; the aggregator verifies the
recorded commit and fail-closed semantics before reporting them.
"""
from __future__ import annotations

import argparse
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


def _load_rows(directory: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[row["scenario_id"]] = row
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-c4l-root", type=Path,
        help="Fresh fail-closed C4L root used to replace S1/S2 legacy shards.",
    )
    args = parser.parse_args(argv)
    out: dict = {"conditions": [], "per_controller_per_condition": {}}
    final_manifest = None
    if args.final_c4l_root:
        args.final_c4l_root = args.final_c4l_root.resolve()
        final_manifest = json.loads(
            (args.final_c4l_root / "run_manifest.json").read_text(encoding="utf-8")
        )
        if final_manifest.get("semantics") != "lazy_exact_fail_closed_no_fallback":
            raise RuntimeError("Final C4L root does not declare fail-closed semantics")
    for cond in sorted(SENS.iterdir()):
        if not cond.is_dir():
            continue
        out["conditions"].append(cond.name)
        per_ctrl: dict = {}
        for ctrl in sorted(cond.iterdir()):
            if not ctrl.is_dir():
                continue
            source_dir = ctrl
            if args.final_c4l_root and cond.name in {"S1", "S2"} and ctrl.name == "C4L":
                source_dir = args.final_c4l_root / cond.name / ctrl.name
            walls = []
            completes = 0
            invariants = 0
            overruns = 0
            infeasibles = 0
            realized_B_values = []
            budget_values = []
            scenario_ids = set()
            for f in source_dir.glob("*.json"):
                try:
                    j = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                scenario_ids.add(j.get("scenario_id"))
                if source_dir != ctrl:
                    metadata = j.get("evaluation_metadata") or {}
                    if metadata.get("semantics") != final_manifest["semantics"]:
                        raise RuntimeError(f"Unexpected semantics in {f}")
                    if metadata.get("repository_commit") != final_manifest["repository_commit"]:
                        raise RuntimeError(f"Mixed repository commit in {f}")
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
                "n_unique_scenarios": len(scenario_ids),
                "source": str(source_dir.relative_to(V108)),
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
    if args.final_c4l_root:
        audit = {
            "source": str(args.final_c4l_root.relative_to(V108)),
            "manifest": final_manifest,
            "conditions": {},
        }
        for cond in ("S1", "S2"):
            c4 = _load_rows(args.final_c4l_root / cond / "C4L")
            c5 = _load_rows(SENS / cond / "C5")
            if len(c4) != 128 or len(c5) != 128 or set(c4) != set(c5):
                raise RuntimeError(f"Incomplete paired C4L/C5 sensitivity data for {cond}")
            complete = {sid for sid, row in c4.items() if row.get("completion", False)}
            infeasible = set(c4) - complete
            c5_overrun = {
                sid for sid, row in c5.items()
                if float(row.get("realized_episode_B_ml", 0.0))
                > float(row.get("budget_ml", 0.0)) + 1e-9
            }
            audit["conditions"][cond] = {
                "n": len(c4),
                "c4l_complete": len(complete),
                "c4l_infeasible": len(infeasible),
                "c4l_overrun": sum(
                    float(row.get("realized_episode_B_ml", 0.0))
                    > float(row.get("budget_ml", 0.0)) + 1e-9
                    for row in c4.values()
                ),
                "infeasible_at_zero_macro_actions": sum(
                    int(c4[sid].get("macro_action_count", -1)) == 0 for sid in infeasible
                ),
                "c5_overrun": len(c5_overrun),
                "c5_overrun_and_c4l_infeasible": len(c5_overrun & infeasible),
                "c5_in_budget_and_c4l_infeasible": len(infeasible - c5_overrun),
            }
        out["final_failclosed_audit"] = audit
        out["notes"] = (
            "S1/S2 C4L results come from a fresh, isolated 128-scene run with "
            "uniform lazy-exact fail-closed semantics; legacy mixed shards are not summarized."
        )
    else:
        out["notes"] = (
            "No final C4L root supplied; S1/S2 may contain legacy mixed-semantics shards."
        )
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
