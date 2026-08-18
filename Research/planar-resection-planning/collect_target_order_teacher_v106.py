"""Collect corrected full-candidate tail labels on v10.6 policy_train only."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from benchmark_target_order_v105 import _tail_worker_v105
from clinical_macro_environment import (
    CLINICAL_MACRO_OBSERVATION_CHANNELS, ClinicalMacroResectionEnv,
)
from clinical_target_order_features_v106 import (
    CANDIDATE_FEATURE_DIM, GLOBAL_FEATURE_DIM,
    candidate_features_v106, compute_feature_scales, global_context_v106,
)
from plan_target_order_v104 import _step_macro_target, serpentine_target_of
from plan_target_order_v105 import (
    DEFAULT_GATE_CLINICAL_CONFIG, SerpentineTailV105,
    _candidate_sources_v105, _env_state_payload_v105,
)

BASE = SIM / "results/clinical_window_v10_6_shielded_learning"
FROZEN = BASE / "frozen"
AUTHORIZED_SCENES = FROZEN / "split_policy_train.json"
AUTHORIZED_BASELINES = FROZEN / "baseline_policy_train.json"
SOURCE_CODE = {"s_target": 0, "exposed": 1, "near_hidden": 2, "nearest": 3, "fill": 4}
CFG = dict(DEFAULT_GATE_CLINICAL_CONFIG)
BINARY_GRID_CHANNELS = (
    "domain", "cut", "hidden_vessel", "exposed_vessel", "sealed_vessel",
    "frontier", "large_vessel", "current_position", "previous_position", "start",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_fingerprint(env: ClinicalMacroResectionEnv) -> str:
    key = SerpentineTailV105()._state_key(env)
    return hashlib.sha256(repr(key).encode()).hexdigest()


def collect_scene(scene, baseline_blood, margin, blood_scale, leaf_pool, out_path: Path):
    env = ClinicalMacroResectionEnv(scenario=scene, clinical_config=CFG, mechanics_update_interval=0)
    env.reset()
    budget = float(baseline_blood) + float(margin)
    states = []
    chosen_actions = []
    invariant_count = s_missing = terminal_conflicts = 0
    while not env.terminated and not env.truncated:
        sourced = _candidate_sources_v105(env, count=6, transfer_counts=env._transfer_counts())
        if not sourced:
            raise RuntimeError(f"no candidates in nonterminal state {scene['scenario_id']}")
        s_target = serpentine_target_of(env)
        s_missing += int(s_target not in [target for target, _ in sourced])
        before_t = float(env.elapsed_minutes)
        before_b = float(env.expected_blood_loss_ml)
        gc, raw_budget = global_context_v106(
            env, baseline_blood_ml=baseline_blood, margin_ml=margin,
            blood_scale_ml=blood_scale,
        )
        observation = env._observation()
        channel_index = {name: i for i, name in enumerate(CLINICAL_MACRO_OBSERVATION_CHANNELS)}
        binary_grid = np.stack([
            observation[channel_index[name]] > 0.5 for name in BINARY_GRID_CHANNELS
        ]).reshape(len(BINARY_GRID_CHANNELS), -1)
        grid_bits = np.packbits(binary_grid, axis=1)
        transfer_q = np.rint(np.clip(observation[channel_index["transfer_distance"]], 0.0, 1.0) * 255.0).astype(np.uint8)
        payload = _env_state_payload_v105(env)
        work = [(scene, payload, target, CFG) for target, _ in sourced]
        future = leaf_pool.map(_tail_worker_v105, work) if leaf_pool else [
            _tail_worker_v105(item) for item in work
        ]
        candidates = []
        for (target, source), (future_t, future_b, completion, reason) in zip(sourced, future):
            feat, after, dt_action, db_action, meta = candidate_features_v106(
                env, target, source=source
            )
            t_tail = float(future_t) - dt_action
            b_tail = float(future_b) - db_action
            t_total = before_t + float(future_t)
            b_total = before_b + float(future_b)
            if after.terminated and after.failure_reason is None and not completion:
                terminal_conflicts += 1
            candidates.append({
                "target": target, "source": source, "feature": feat,
                "dt_action": dt_action, "db_action": db_action,
                "t_tail": t_tail, "b_tail": b_tail,
                "t_total": t_total, "b_total": b_total,
                "completion": bool(completion), "failure_reason": reason,
                "safe": bool(completion and reason is None and b_total <= budget + 1e-9),
                "is_s": bool(meta["is_serpentine_fallback"]),
            })
        safe = [candidate for candidate in candidates if candidate["safe"]]
        if not safe:
            invariant_count += 1
            chosen = next(candidate for candidate in candidates if candidate["target"] == s_target)
        else:
            chosen = min(
                safe,
                key=lambda item: (
                    round(item["t_total"], 6), round(item["b_total"], 6), item["target"]
                ),
            )
        states.append({
            "global": gc, "raw_budget": raw_budget,
            "grid_bits": grid_bits, "transfer_q": transfer_q,
            "fingerprint": state_fingerprint(env), "step": int(env.step_count),
            "candidates": candidates,
        })
        chosen_actions.append(chosen["target"])
        _step_macro_target(env, chosen["target"])

    n = len(states); k = 6
    arrays = {
        "features": np.zeros((n, k, CANDIDATE_FEATURE_DIM), np.float32),
        "global_context": np.zeros((n, GLOBAL_FEATURE_DIM), np.float32),
        "valid": np.zeros((n, k), bool),
        "targets": np.full((n, k, 2), -1, np.int32),
        "source": np.full((n, k), -1, np.int8),
        "is_s": np.zeros((n, k), bool),
        "delta_T_action": np.full((n, k), np.nan, np.float64),
        "delta_B_action": np.full((n, k), np.nan, np.float64),
        "T_tail": np.full((n, k), np.nan, np.float64),
        "B_tail": np.full((n, k), np.nan, np.float64),
        "T_total": np.full((n, k), np.nan, np.float64),
        "B_total": np.full((n, k), np.nan, np.float64),
        "completion": np.zeros((n, k), bool),
        "safe_exact": np.zeros((n, k), bool),
        "B_past": np.zeros(n, np.float64),
        "B_baseline_scene": np.full(n, baseline_blood, np.float64),
        "M_B": np.full(n, margin, np.float64),
        "B_budget_total": np.full(n, budget, np.float64),
        "B_remaining": np.zeros(n, np.float64),
        "state_step": np.zeros(n, np.int32),
        "state_fingerprint": np.empty(n, dtype="U64"),
        "grid_bits": np.zeros((n, len(BINARY_GRID_CHANNELS), 150), np.uint8),
        "transfer_q": np.zeros((n, 30, 40), np.uint8),
    }
    for si, state in enumerate(states):
        arrays["global_context"][si] = state["global"]
        arrays["B_past"][si] = state["raw_budget"]["B_past_ml"]
        arrays["B_remaining"][si] = state["raw_budget"]["B_remaining_ml"]
        arrays["state_step"][si] = state["step"]
        arrays["state_fingerprint"][si] = state["fingerprint"]
        arrays["grid_bits"][si] = state["grid_bits"]
        arrays["transfer_q"][si] = state["transfer_q"]
        for ci, candidate in enumerate(state["candidates"]):
            arrays["features"][si, ci] = candidate["feature"]
            arrays["valid"][si, ci] = True
            arrays["targets"][si, ci] = candidate["target"]
            arrays["source"][si, ci] = SOURCE_CODE[candidate["source"]]
            arrays["is_s"][si, ci] = candidate["is_s"]
            for key, source_key in (
                ("delta_T_action", "dt_action"), ("delta_B_action", "db_action"),
                ("T_tail", "t_tail"), ("B_tail", "b_tail"),
                ("T_total", "t_total"), ("B_total", "b_total"),
                ("completion", "completion"), ("safe_exact", "safe"),
            ):
                arrays[key][si, ci] = candidate[source_key]
    arrays["scenario_id"] = np.asarray([scene["scenario_id"]] * n, dtype="U64")
    arrays["realized_T"] = np.asarray([env.elapsed_minutes], np.float64)
    arrays["realized_B"] = np.asarray([env.expected_blood_loss_ml], np.float64)
    arrays["action_hash"] = np.asarray([
        hashlib.sha256(json.dumps(chosen_actions, separators=(",", ":")).encode()).hexdigest()
    ], dtype="U64")
    arrays["invariant_count"] = np.asarray([invariant_count], np.int32)
    arrays["s_missing"] = np.asarray([s_missing], np.int32)
    arrays["terminal_conflicts"] = np.asarray([terminal_conflicts], np.int32)
    arrays["completion_scene"] = np.asarray([
        bool(env.terminated and env.failure_reason is None)
    ], bool)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".npz.partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(out_path)
    return {
        "scenario_id": scene["scenario_id"], "states": n,
        "candidates": int(arrays["valid"].sum()),
        "invariant_count": invariant_count, "s_missing": s_missing,
        "terminal_conflicts": terminal_conflicts,
        "completion": bool(arrays["completion_scene"][0]),
        "realized_T": float(env.elapsed_minutes), "realized_B": float(env.expected_blood_loss_ml),
        "shard": out_path.name, "sha256": sha256(out_path),
    }


def audit_from_shard(path: Path) -> dict:
    data = np.load(path)
    row = {
        "scenario_id": str(data["scenario_id"][0]),
        "states": int(len(data["global_context"])),
        "candidates": int(data["valid"].sum()),
        "invariant_count": int(data["invariant_count"][0]),
        "s_missing": int(data["s_missing"][0]),
        "terminal_conflicts": int(data["terminal_conflicts"][0]),
        "completion": bool(data["completion_scene"][0]),
        "realized_T": float(data["realized_T"][0]),
        "realized_B": float(data["realized_B"][0]),
        "shard": path.name,
        "sha256": sha256(path),
    }
    data.close()
    return row


def _child(task_queue, result_queue, margin, blood_scale, leaf_workers, shard_dir):
    pool = mp.get_context("fork").Pool(leaf_workers)
    try:
        while True:
            item = task_queue.get()
            if item is None:
                break
            scene, baseline = item
            path = Path(shard_dir) / f"{scene['scenario_id']}.npz"
            try:
                row = collect_scene(
                    scene, baseline, margin, blood_scale, pool, path
                )
                print(f"teacher shard {row['scenario_id']} states={row['states']}", flush=True)
                result_queue.put((row, None))
            except BaseException as exc:
                result_queue.put((None, f"{scene['scenario_id']}: {exc!r}"))
    finally:
        pool.close(); pool.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-workers", type=int, default=8)
    parser.add_argument("--leaf-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=BASE / "teacher")
    args = parser.parse_args()
    scenes_payload = json.loads(AUTHORIZED_SCENES.read_text(encoding="utf-8"))
    base_payload = json.loads(AUTHORIZED_BASELINES.read_text(encoding="utf-8"))
    if scenes_payload["split"] != "policy_train" or base_payload["split"] != "policy_train":
        raise RuntimeError("collector authorization failure")
    scenes = scenes_payload["scenarios"]
    if args.limit:
        scenes = scenes[:args.limit]
    baselines = base_payload["records"]
    scales = json.loads((FROZEN / "scales_v10_6.json").read_text(encoding="utf-8"))
    margin = float(scales["margin_ml"]); blood_scale = float(scales["blood_scale_ml"])
    shard_dir = args.output_dir / "shards"
    final_audit = args.output_dir / "teacher_data_audit.json"
    if final_audit.exists() and not args.limit:
        raise FileExistsError(f"Refusing to overwrite completed teacher directory {args.output_dir}")
    existing_audits = []
    if shard_dir.exists() and not args.limit:
        for path in sorted(shard_dir.glob("*.npz")):
            existing_audits.append(audit_from_shard(path))
    existing_ids = {row["scenario_id"] for row in existing_audits}
    tasks = [(scene, float(baselines[scene["scenario_id"]]["expected_blood_loss_ml"]))
             for scene in scenes if scene["scenario_id"] not in existing_ids]
    worker_count = min(args.scene_workers, len(tasks))
    task_queue = mp.get_context("fork").Queue()
    result_queue = mp.get_context("fork").Queue()
    processes = []; wall0 = time.time()
    for _ in range(worker_count):
        process = mp.get_context("fork").Process(
            target=_child,
            args=(task_queue, result_queue, margin, blood_scale,
                  args.leaf_workers, shard_dir),
        )
        process.start(); processes.append(process)
    for task in tasks:
        task_queue.put(task)
    for _ in processes:
        task_queue.put(None)
    audits = list(existing_audits); errors = []
    for _ in tasks:
        row, error = result_queue.get()
        if row is not None: audits.append(row)
        if error: errors.append(error)
    for process in processes:
        process.join()
        if process.exitcode: errors.append(f"worker exitcode={process.exitcode}")
    if errors:
        raise RuntimeError("; ".join(errors))
    if len(audits) != len(scenes):
        raise RuntimeError(f"teacher shard count mismatch: {len(audits)} != {len(scenes)}")
    audits.sort(key=lambda row: row["scenario_id"])

    # Consolidate immutable per-scene shards into the training artifact.
    shard_data = [np.load(shard_dir / row["shard"]) for row in audits]
    state_keys = [key for key in shard_data[0].files if key not in {
        "realized_T", "realized_B", "action_hash", "invariant_count",
        "s_missing", "terminal_conflicts", "completion_scene",
    }]
    merged = {key: np.concatenate([data[key] for data in shard_data], axis=0) for key in state_keys}
    output = args.output_dir / "teacher_rankings_v106.npz"
    np.savez_compressed(output, **merged)
    feature_scales = compute_feature_scales(merged["features"][merged["valid"]])
    (args.output_dir / "feature_scales_v106.json").write_text(
        json.dumps(feature_scales, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    valid = merged["valid"]
    identity_error = np.max(np.abs(
        merged["B_total"][valid] - (
            np.repeat(merged["B_past"][:, None], valid.shape[1], axis=1)[valid]
            + merged["delta_B_action"][valid] + merged["B_tail"][valid]
        )
    ))
    audit = {
        "version": "v10.6-teacher-data-audit-v1",
        "authorized_split_files": [AUTHORIZED_SCENES.name, AUTHORIZED_BASELINES.name],
        "binary_grid_channels": list(BINARY_GRID_CHANNELS),
        "transfer_quantization": "round(clip(transfer_distance,0,1)*255)",
        "scene_count": len(audits), "state_count": int(valid.shape[0]),
        "candidate_count": int(valid.sum()),
        "completion_failures": sum(not row["completion"] for row in audits),
        "invariant_count": sum(row["invariant_count"] for row in audits),
        "s_missing": sum(row["s_missing"] for row in audits),
        "terminal_conflicts": sum(row["terminal_conflicts"] for row in audits),
        "B_total_identity_max_abs_error": float(identity_error),
        "safe_candidate_fraction": float(merged["safe_exact"][valid].mean()),
        "unsafe_candidate_count": int((~merged["safe_exact"][valid]).sum()),
        "B_tail_p50_p95_p99_max": [float(x) for x in np.quantile(
            merged["B_tail"][valid], [0.5, 0.95, 0.99, 1.0]
        )],
        "B_total_p50_p95_p99_max": [float(x) for x in np.quantile(
            merged["B_total"][valid], [0.5, 0.95, 0.99, 1.0]
        )],
        "teacher_npz": output.name, "teacher_npz_sha256": sha256(output),
        "wall_seconds": time.time() - wall0,
        "scenes": audits,
    }
    (args.output_dir / "teacher_data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: audit[key] for key in (
        "scene_count", "state_count", "candidate_count", "completion_failures",
        "invariant_count", "s_missing", "terminal_conflicts",
        "B_total_identity_max_abs_error", "wall_seconds",
    )}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
