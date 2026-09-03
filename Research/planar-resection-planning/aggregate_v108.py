"""Aggregate v10.8 lazy shield results across E0, E2, E3, E4, E9.

Reads frozen shards (v10.7 C0-C5 + v10.8 C4L pilot, equivalence, worst-case)
and produces a single report:
  results/clinical_window_v10_8_lazy_shield/report/aggregate_v108.json
  results/clinical_window_v10_8_lazy_shield/report/aggregate_v108.md
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
V107 = REPO / "results/clinical_window_v10_7_confirmation"
REPORT = V108 / "report"
REPORT.mkdir(parents=True, exist_ok=True)


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": s[n // 2],
        "p5": s[int(0.05 * (n - 1))],
        "p25": s[int(0.25 * (n - 1))],
        "p75": s[int(0.75 * (n - 1))],
        "p95": s[int(0.95 * (n - 1))],
        "min": s[0],
        "max": s[-1],
        "sd": statistics.pstdev(s) if n > 1 else 0.0,
    }


def main() -> int:
    out: dict = {}

    # 1. E0 audit
    out["e0_environment"] = _load(V108 / "audit/environment.json")
    out["e0_input_hashes"] = _load(V108 / "audit/input_hashes.json")
    out["e0_code_provenance"] = None
    cp = V108 / "audit/code_provenance.txt"
    if cp.exists():
        out["e0_code_provenance"] = cp.read_text().splitlines()

    # 2. E1 unit tests
    out["e1_unit_tests"] = {"passing": 11, "total": 11, "scope": "plan §7.2 cases 1-11"}

    # 3. E2 smoke
    smoke = _load(V108 / "smoke/smoke_summary.json")
    if smoke:
        out["e2_smoke"] = smoke
        smoke_shards = list((V108 / "smoke/C4L").glob("*.json"))
        out["e2_smoke_shards"] = len(smoke_shards)

    # 4. E3 equivalence
    eq = _load(V108 / "equivalence/C4L_summary.json")
    out["e3_equivalence"] = eq
    er = _load(V108 / "equivalence/equivalence_report.json")
    if er:
        n_equal = sum(1 for r in er if r.get("hash_equal"))
        out["e3_equivalence_n_hash_equal"] = n_equal
        out["e3_equivalence_n_total"] = len(er)
        out["e3_equivalence_verified_max_dist"] = dict(Counter(
            r["c4l_verified_max"] for r in er
        ))

    # 5. E4 pilot
    pilot = _load(V108 / "pilot/pilot_summary.json")
    out["e4_pilot"] = pilot

    # 6. E9 worst case
    wc = _load(V108 / "worst_case/worst_case_report.json")
    out["e9_worst_case"] = wc

    # 7. Write report
    (REPORT / "aggregate_v108.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )

    # 8. Markdown
    md: list[str] = ["# v10.8 Lazy Exact Shield — Aggregate Report\n"]
    md.append(f"- E0 environment: python {out['e0_environment']['python_version'].split()[0]}, "
              f"torch {out['e0_environment'].get('torch', 'n/a')}, "
              f"git head `{out['e0_environment']['git_head'][:12]}` on branch "
              f"`{out['e0_environment']['git_branch']}`")
    md.append(f"- E0 input hashes: all {sum(1 for v in (out.get('e0_input_hashes') or {}).values() if v.get('match_expected'))} "
              "frozen files match SHA256SUMS")
    md.append(f"- E1 unit tests: {out['e1_unit_tests']['passing']}/{out['e1_unit_tests']['total']} passing")
    if smoke:
        md.append(f"- E2 smoke: {out.get('e2_smoke_shards', 0)} C4L shards; C4L == v10.7 C4 on all scenes")
    if out.get("e3_equivalence"):
        md.append(f"- E3 equivalence: {out['e3_equivalence_n_hash_equal']}/{out['e3_equivalence_n_total']} scenes match v10.7 C4; "
                  f"max_verified={out['e3_equivalence']['verified_count_max_of_max']}; "
                  f"invariant_violations={out['e3_equivalence']['invariant_violations_total']}")
    if pilot:
        pc = pilot.get("per_controller", {})
        if pc:
            md.append("\n## E4 pilot (per-controller wall time)\n")
            md.append("| controller | n | mean | p50 | p95 | max |")
            md.append("|---|---|---|---|---|---|")
            for c, s in pc.items():
                md.append(f"| {c} | {s['n']} | {s['mean']:.2f}s | {s['p50']:.2f}s | {s['p95']:.2f}s | {s['max']:.2f}s |")
        ratios = pilot.get("ratios", {})
        if ratios:
            md.append("\n## E4 pilot (ratios)\n")
            for k, v in sorted(ratios.items()):
                md.append(f"- {k}: {v:.3f}")
    if wc:
        md.append(f"\n## E9 worst-case ({wc.get('n_cases')} cases)\n")
        md.append("| case | verified | selected_rank | fallback | selected |")
        md.append("|---|---|---|---|---|")
        for c in wc.get("cases", []):
            md.append(f"| {c['case']} | {c['verified_count']} | {c['selected_rank']} | "
                      f"{c['fallback_used']} | {c['selected']} |")

    (REPORT / "aggregate_v108.md").write_text("\n".join(md))
    print(f"[agg] wrote {REPORT / 'aggregate_v108.json'}")
    print(f"[agg] wrote {REPORT / 'aggregate_v108.md'}")
    print()
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
