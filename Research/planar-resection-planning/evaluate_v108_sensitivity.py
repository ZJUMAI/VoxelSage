"""E8 sensitivity: per-(controller, condition) phase.

Reuses the in-worker shard write pattern of ``evaluate_v108_phase``.
Each (controller, condition) gets its own output directory so the
sensitivity shards do not collide with the main E6 shards.

Usage:
  evaluate_v108_sensitivity.py --controllers C0,C3,C4E,C4L,C5 \\
                                --conditions S0,S1,S2,S3,S4 \\
                                --scene-workers 20 --limit 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

V108_OUT = REPO / "results/clinical_window_v10_8_lazy_shield"
SHARDS_OUT = V108_OUT / "sensitivity"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

SENSITIVITY_CONDITIONS = {
    "S0": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S1": {"max_clamp_minutes": 12.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S2": {"max_clamp_minutes": 10.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S3": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.5},
    "S4": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.25},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _task(args):
    controller, cond, sid, scene, baseline, margin, ckpt, cfg, out_dir, metadata = args
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sid}.json"
    if out.exists():
        return sid, controller, cond, "skip", 0.0
    t0 = time.time()
    try:
        from lazy_confirmation_controllers_v108 import rollout_controller
        res = rollout_controller(
            controller, scene,
            baseline_blood=float(baseline), margin_ml=float(margin),
            cfg=cfg, checkpoint_path=str(ckpt),
        )
        shard = dict(res)
        shard["scenario_id"] = sid
        shard["controller"] = controller
        shard["condition"] = cond
        shard["evaluation_metadata"] = metadata
        tmp = out.with_suffix(f"{out.suffix}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)
        return sid, controller, cond, "ok", time.time() - t0
    except BaseException as e:
        return sid, controller, cond, f"err:{e!r}", time.time() - t0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path,
                        default=V108_OUT / "frozen/split_lazy_replication.json")
    parser.add_argument("--baseline-file", type=Path,
                        default=V108_OUT / "frozen/baseline_lazy_replication.json")
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--scene-workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N scenarios of the split; used to "
                             "extend the 64-scene sensitivity subset to 128 "
                             "scenes without re-running the first 64.")
    parser.add_argument("--controllers", default="C0,C3,C4E,C4L,C5")
    parser.add_argument("--conditions", default="S0,S1,S2,S3,S4")
    parser.add_argument(
        "--output-root", type=Path, default=SHARDS_OUT,
        help="Independent shard root; use a fresh directory for a new semantics audit.",
    )
    args = parser.parse_args(argv)
    controllers = [c for c in args.controllers.split(",") if c]
    conds = [c for c in args.conditions.split(",") if c]

    args.output_root.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = split["scenarios"][args.offset:args.offset + args.limit]
    margin = args.margin
    metadata = {
        "semantics": "lazy_exact_fail_closed_no_fallback",
        "repository_commit": _repository_commit(),
        "runner": Path(__file__).name,
        "split_sha256": _sha256(args.split_file),
        "baseline_sha256": _sha256(args.baseline_file),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "margin_ml": margin,
    }
    manifest = {
        **metadata,
        "controllers": controllers,
        "conditions": conds,
        "offset": args.offset,
        "limit": args.limit,
        "scene_workers": args.scene_workers,
        "output_root": str(args.output_root.resolve()),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for cond in conds:
        cfg = SENSITIVITY_CONDITIONS[cond]
        for ctrl in controllers:
            out_dir = args.output_root / cond / ctrl
            out_dir.mkdir(parents=True, exist_ok=True)
            tasks = []
            for s in scenes:
                sid = s["scenario_id"]
                if sid not in base["records"]:
                    continue
                out = out_dir / f"{sid}.json"
                if out.exists():
                    continue
                tasks.append((ctrl, cond, sid, s,
                              float(base["records"][sid]["expected_blood_loss_ml"]),
                              margin, str(args.checkpoint), cfg, out_dir, metadata))
            if not tasks:
                print(f"[E8/{cond}/{ctrl}] all done; skip")
                continue
            print(f"[E8/{cond}/{ctrl}] {len(tasks)} task tuples")
            if args.scene_workers <= 1:
                done = 0
                t_start = time.time()
                for t in tasks:
                    sid, c, cond, status, _ = _task(t)
                    done += 1
                    if done % 50 == 0:
                        print(f"  done {done}/{len(tasks)} ({(time.time()-t_start)/60:.1f} min)")
            else:
                ctx = mp.get_context()
                done = 0
                t_start = time.time()
                with ctx.Pool(args.scene_workers) as pool:
                    for sid, c, cond, status, _ in pool.imap_unordered(_task, tasks, chunksize=4):
                        done += 1
                        if done % 50 == 0:
                            elapsed = time.time() - t_start
                            rate = done / elapsed
                            eta = (len(tasks) - done) / rate
                            print(f"  done {done}/{len(tasks)} ({rate:.2f}/s, ETA {eta/60:.1f} min)")
                print(f"  total {cond}/{ctrl}: {done} done in {(time.time()-t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
