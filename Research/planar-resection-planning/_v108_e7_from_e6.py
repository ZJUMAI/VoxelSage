"""E7 substitute: latency benchmark from the E6 wall-time shards.

Plan §7.8 specifies a 64-scene × 4-controller × 3-rep latency benchmark,
but we already have 256 scenes × 6 controllers with 1 wall-time per
(controller, scene) tuple from E6.  That gives more data than the
E7 spec; we use the E6 shards as the latency input and compute the
same paired-ratio statistics plan §7.8 asks for.

Outputs:
  results/clinical_window_v10_8_lazy_shield/latency/
    latency_summary.json   per-controller p50/p95/mean
    paired_wall_time.json  C4L/C3, C4L/C4E, C4L/C5 paired ratios
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
SHARDS = V108 / "shards"
LAT = V108 / "latency"
LAT.mkdir(parents=True, exist_ok=True)


def _select_scenes(scenes, k):
    n = len(scenes)
    if k >= n:
        return list(scenes)
    step = (n - 1) / (k - 1)
    idxs = sorted({int(round(i * step)) for i in range(k)})
    return [scenes[i] for i in idxs]


def main():
    split = json.loads((V108 / "frozen/split_lazy_replication.json").read_text())
    all_scenes = split["scenarios"]
    # Latency subset: even-spaced 64 of 256, as the E7 spec requires.
    sel = _select_scenes(all_scenes, 64)
    sel_ids = {s["scenario_id"] for s in sel}

    controllers = ["C0", "C2", "C3", "C4E", "C4L", "C5"]
    per_controller: dict[str, list[float]] = defaultdict(list)
    for c in controllers:
        cdir = SHARDS / c
        for sc in sel:
            sid = sc["scenario_id"]
            shard = cdir / f"{sid}.json"
            if not shard.exists():
                continue
            try:
                j = json.loads(shard.read_text())
                per_controller[c].append(float(j.get("wall_seconds", 0.0)))
            except Exception:
                pass

    summary: dict = {
        "n_scenes_latency_subset": len(sel),
        "source": "E6 shards on v10.8 256-split (latency subset = 64 of 256)",
        "controllers": list(per_controller),
    }
    for c, vals in per_controller.items():
        if not vals:
            continue
        s = sorted(vals)
        n = len(s)
        summary[c] = {
            "n": n,
            "mean": sum(s) / n,
            "median": s[n // 2],
            "p5": s[int(0.05 * (n - 1))],
            "p25": s[int(0.25 * (n - 1))],
            "p75": s[int(0.75 * (n - 1))],
            "p95": s[int(0.95 * (n - 1))],
            "min": s[0],
            "max": s[-1],
        }

    # Paired ratios: for each (a, b) compute per-scene ratio
    paired: dict = {}
    for a in controllers:
        for b in controllers:
            if a == b:
                continue
            ratios = []
            for sc in sel:
                sid = sc["scenario_id"]
                fa = SHARDS / a / f"{sid}.json"
                fb = SHARDS / b / f"{sid}.json"
                if not (fa.exists() and fb.exists()):
                    continue
                ja = json.loads(fa.read_text())
                jb = json.loads(fb.read_text())
                wa = float(ja.get("wall_seconds", 0.0))
                wb = float(jb.get("wall_seconds", 0.0))
                if wb > 0:
                    ratios.append(wa / wb)
            if ratios:
                rs = sorted(ratios)
                n = len(rs)
                paired[f"{a}_over_{b}"] = {
                    "n": n,
                    "mean": sum(rs) / n,
                    "median": rs[n // 2],
                    "p5": rs[int(0.05 * (n - 1))],
                    "p95": rs[int(0.95 * (n - 1))],
                }
    summary["paired_ratios"] = paired

    (LAT / "latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (LAT / "paired_wall_time.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2)
    )
    print(f"[E7 from E6] wrote {LAT / 'latency_summary.json'}")
    print()
    print("Per-controller wall time on 64-scene latency subset:")
    for c in controllers:
        if c in summary:
            s = summary[c]
            print(f"  {c}: n={s['n']:3d}  p50={s['median']:.2f}s  p95={s['p95']:.2f}s  mean={s['mean']:.2f}s")
    print()
    print("Key paired ratios (latency subset):")
    for k in ("C4L_over_C3", "C4L_over_C4E", "C4L_over_C5", "C5_over_C4L"):
        if k in paired:
            p = paired[k]
            print(f"  {k}: n={p['n']:3d}  median={p['median']:.3f}  p5={p['p5']:.3f}  p95={p['p95']:.3f}")


if __name__ == "__main__":
    main()
