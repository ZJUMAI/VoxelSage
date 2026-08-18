"""Paired exact-shield rollout for an explicitly authorized v10.6 split file."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_macro_environment import CLINICAL_MACRO_OBSERVATION_CHANNELS, ClinicalMacroResectionEnv
from clinical_safety_shield_v106 import ExactSafetyShieldV106
from clinical_target_order_features_v106 import candidate_features_v106, global_context_v106
from clinical_target_order_policy_v106 import TargetOrderScorerV106
from plan_target_order_v104 import _step_macro_target, serpentine_target_of

BASE = SIM / "results/clinical_window_v10_6_shielded_learning"
FROZEN = BASE / "frozen"
CFG = {"early_end_mode": "disabled", "early_end_minutes": 0.0,
       "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}
BINARY_CHANNELS = (
    "domain", "cut", "hidden_vessel", "exposed_vessel", "sealed_vessel",
    "frontier", "large_vessel", "current_position", "previous_position", "start",
)


def bootstrap_ci(values, seed, samples=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[index].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def model_inputs(env, records, checkpoint):
    channel = {name: i for i, name in enumerate(CLINICAL_MACRO_OBSERVATION_CHANNELS)}
    obs = env._observation()
    grid = np.stack([obs[channel[name]] for name in BINARY_CHANNELS] +
                    [obs[channel["transfer_distance"]]]).astype(np.float32)
    source = {record.target: record.source for record in records}
    features = np.stack([
        candidate_features_v106(env, record.target, source=source[record.target])[0]
        for record in records
    ]).astype(np.float32)
    scales = checkpoint["feature_scales"]
    mean = np.asarray(scales["mean"], np.float32)
    std = np.asarray(scales["std"], np.float32)
    idx = np.asarray(scales["scaled_indices"], dtype=int)
    features[:, idx] = (features[:, idx] - mean[idx]) / std[idx]
    gc, _ = global_context_v106(
        env,
        baseline_blood_ml=env._v106_baseline_blood,
        margin_ml=env._v106_margin,
        blood_scale_ml=checkpoint["blood_scale"],
    )
    targets = np.asarray([record.target for record in records], dtype=np.int64)
    return grid, features, gc.astype(np.float32), targets


def load_model(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model = TargetOrderScorerV106(
        hidden=int(checkpoint["hidden"]), spatial=int(checkpoint["spatial"])
    )
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model, checkpoint


def rollout_one(
    scene, baseline_blood, margin, checkpoint_path, leaf_pool, cache_dir=None,
    include_actions=False,
):
    model, checkpoint = load_model(checkpoint_path)
    env = ClinicalMacroResectionEnv(scenario=scene, clinical_config=CFG, mechanics_update_interval=0)
    env.reset()
    env._v106_baseline_blood = float(baseline_blood)
    env._v106_margin = float(margin)
    budget = float(baseline_blood) + float(margin)
    record_cache = None
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{scene['scenario_id']}.pkl"
        if cache_path.is_file():
            with cache_path.open("rb") as handle:
                record_cache = pickle.load(handle)
        else:
            record_cache = {}
    shield = ExactSafetyShieldV106(
        clinical_config=CFG, leaf_pool=leaf_pool, record_cache=record_cache
    )
    actions = []; interventions = invariants = s_selections = 0
    predicted_safe = predicted_safe_rejected = 0
    selected_max = all_max = 0.0
    worst_candidate_budget_excess = -float("inf")
    forward_ms = shield_ms = 0.0
    wall0 = time.time()
    while not env.terminated and not env.truncated:
        start = time.perf_counter(); records = shield.evaluate(env, budget_ml=budget)
        shield_ms += (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        grid, feature, gc, targets = model_inputs(env, records, checkpoint)
        with torch.no_grad():
            output = model(
                torch.from_numpy(grid[None]), torch.from_numpy(feature[None]),
                torch.from_numpy(gc[None]), torch.from_numpy(targets[None]),
            )
        scores = output["score"][0].numpy()
        predicted_safe_mask = torch.sigmoid(output["safe_logit"][0]).numpy() >= 0.5
        forward_ms += (time.perf_counter() - start) * 1000.0
        safe = [i for i, record in enumerate(records) if record.safe_exact]
        for i, record in enumerate(records):
            if predicted_safe_mask[i]:
                predicted_safe += 1
                predicted_safe_rejected += int(not record.safe_exact)
            worst_candidate_budget_excess = max(
                worst_candidate_budget_excess, float(record.B_total - budget)
            )
        if not safe:
            target = serpentine_target_of(env); invariants += 1
        else:
            top = max(range(len(records)), key=lambda i: (float(scores[i]),
                      -records[i].target[0], -records[i].target[1]))
            chosen_i = max(safe, key=lambda i: (float(scores[i]),
                           -records[i].target[0], -records[i].target[1]))
            chosen = records[chosen_i]; target = chosen.target
            interventions += int(top != chosen_i)
            selected_max = max(selected_max, chosen.B_total)
        s_selections += int(target == serpentine_target_of(env))
        all_max = max(all_max, max(record.B_total for record in records))
        actions.append(target); _step_macro_target(env, target)
    import hashlib
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_suffix(".pkl.partial")
        with partial.open("wb") as handle:
            pickle.dump(record_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        partial.replace(cache_path)
    result = {
        "scenario_id": scene["scenario_id"],
        "completion": bool(env.terminated and env.failure_reason is None),
        "failure_reason": env.failure_reason, "legal_action_rate": 1.0,
        "elapsed_minutes": float(env.elapsed_minutes),
        "realized_episode_B_ml": float(env.expected_blood_loss_ml),
        "budget_ml": budget, "selected_max_B_total_ml": selected_max,
        "all_candidates_max_B_total_ml": all_max,
        "shield_intervention_count": interventions,
        "s_selection_count": s_selections,
        "model_predicted_safe_count": predicted_safe,
        "model_predicted_safe_but_shield_rejected_count": predicted_safe_rejected,
        "worst_candidate_B_total_minus_budget_ml": worst_candidate_budget_excess,
        "safety_invariant_violations": invariants,
        "macro_action_count": len(actions), "policy_forward_ms": forward_ms,
        "shield_exact_ms": shield_ms, "wall_seconds": time.time() - wall0,
        "shield_record_cache_hits": shield.record_cache_hits,
        "shield_record_cache_misses": shield.record_cache_misses,
        "action_sequence_hash": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if include_actions:
        result["actions"] = [[int(row), int(col)] for row, col in actions]
    return result


def _worker(
    task_queue, result_queue, checkpoint_path, margin, leaf_workers, cache_dir,
    include_actions,
):
    torch.set_num_threads(1)
    pool = mp.get_context("fork").Pool(leaf_workers)
    try:
        while True:
            task = task_queue.get()
            if task is None: break
            scene, baseline = task
            try:
                result_queue.put((rollout_one(
                    scene, baseline, margin, checkpoint_path, pool, cache_dir,
                    include_actions,
                ), None))
            except BaseException as exc:
                result_queue.put((None, f"{scene['scenario_id']}: {exc!r}"))
    finally:
        pool.close(); pool.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--teacher-gate", type=Path, default=BASE / "evaluation/gate_t_teacher.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-workers", type=int, default=8)
    parser.add_argument("--leaf-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument(
        "--shield-cache-dir", type=Path, default=None,
        help="Per-scene exact-state record cache; affects runtime only, never policy scores or shield semantics.",
    )
    parser.add_argument(
        "--include-actions", action="store_true",
        help="Store the frozen target sequence for mechanics-only replay.",
    )
    args = parser.parse_args()
    split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))
    baseline_payload = json.loads(args.baseline_file.read_text(encoding="utf-8"))
    if split_payload["split"] != baseline_payload["split"]:
        raise RuntimeError("scene/baseline split mismatch")
    split_name = split_payload["split"]
    scenes = split_payload["scenarios"]
    if args.scenario_id is not None:
        scenes = [scene for scene in scenes if scene["scenario_id"] == args.scenario_id]
        if len(scenes) != 1:
            raise RuntimeError(f"scenario-id must resolve exactly once: {args.scenario_id}")
    if args.limit: scenes = scenes[:args.limit]
    baselines = baseline_payload["records"]
    scales = json.loads((FROZEN / "scales_v10_6.json").read_text(encoding="utf-8"))
    manifest = json.loads((FROZEN / "experiment_manifest.json").read_text(encoding="utf-8"))
    margin = float(scales["margin_ml"])
    tasks = [(scene, float(baselines[scene["scenario_id"]]["expected_blood_loss_ml"])) for scene in scenes]
    workers = min(args.scene_workers, len(tasks)); task_q = mp.get_context("fork").Queue()
    result_q = mp.get_context("fork").Queue(); processes = []
    for _ in range(workers):
        p = mp.get_context("fork").Process(
            target=_worker, args=(task_q, result_q, args.checkpoint, margin,
                                  args.leaf_workers, args.shield_cache_dir,
                                  args.include_actions)
        ); p.start(); processes.append(p)
    for task in tasks: task_q.put(task)
    for _ in processes: task_q.put(None)
    rows = []; errors = []; wall0 = time.time()
    for _ in tasks:
        row, error = result_q.get()
        if row: rows.append(row)
        if error: errors.append(error)
    for p in processes: p.join()
    if errors: raise RuntimeError("; ".join(errors))
    rows.sort(key=lambda row: row["scenario_id"])
    d_t = []; d_b = []
    for row in rows:
        base = baselines[row["scenario_id"]]
        row["baseline_T_min"] = float(base["elapsed_minutes"])
        row["baseline_B_ml"] = float(base["expected_blood_loss_ml"])
        row["delta_T_min"] = row["elapsed_minutes"] - row["baseline_T_min"]
        row["delta_B_ml"] = row["realized_episode_B_ml"] - row["baseline_B_ml"]
        d_t.append(row["delta_T_min"]); d_b.append(row["delta_B_ml"])
    seed = int(manifest["bootstrap_seed"])
    teacher = json.loads(args.teacher_gate.read_text(encoding="utf-8")) if args.teacher_gate.is_file() else None
    teacher_gain = None
    if teacher:
        teacher_ids = {row["scenario_id"] for row in teacher.get("rows", [])}
        policy_ids = {row["scenario_id"] for row in rows}
        if teacher_ids != policy_ids:
            raise RuntimeError("teacher reference scenario IDs do not match evaluated policy split")
        base_t = float(np.mean([r["baseline_T_min"] for r in rows]))
        policy_t = float(np.mean([r["elapsed_minutes"] for r in rows]))
        teacher_t = float(teacher["summary"]["mean_teacher_T_min"])
        denominator = base_t - teacher_t
        if denominator <= 0:
            raise RuntimeError("teacher reference has no positive time benefit")
        teacher_gain = (base_t - policy_t) / denominator
    overrun = sum(value > margin + 1e-9 for value in d_b)
    failures = sum(
        bool((not row["completion"]) or (row["failure_reason"] is not None))
        for row in rows
    )
    invariants = sum(row["safety_invariant_violations"] for row in rows)
    ci_t = bootstrap_ci(d_t, seed); ci_b = bootstrap_ci(d_b, seed)
    conditions = {
        "completion_legal_100": failures == 0,
        "failure_invariant_zero": failures == 0 and invariants == 0,
        "per_scene_overrun_zero": overrun == 0,
        "max_delta_B_le_margin": max(d_b) <= margin + 1e-9,
        "delta_B_ci_upper_le_margin": ci_b[1] <= margin,
        "delta_T_ci_upper_lt_zero": ci_t[1] < 0,
        "teacher_benefit_retention_ge_050": teacher_gain is None or teacher_gain >= 0.5,
    }
    result = {
        "version": "v10.6-shielded-policy-evaluation-v1", "split": split_name,
        "checkpoint": str(args.checkpoint), "n_scenarios": len(rows), "margin_ml": margin,
        "conditions": conditions, "decision": "GO" if all(conditions.values()) else "NO-GO",
        "summary": {
            "failures": failures, "invariants": invariants, "overrun_count": overrun,
            "max_delta_B_ml": max(d_b), "mean_delta_B_ml": float(np.mean(d_b)),
            "delta_B_95_ci": ci_b, "mean_delta_T_min": float(np.mean(d_t)),
            "delta_T_95_ci": ci_t, "teacher_benefit_retention": teacher_gain,
            "shield_intervention_action_rate": float(sum(r["shield_intervention_count"] for r in rows)
                                                       / max(1, sum(r["macro_action_count"] for r in rows))),
            "shield_intervention_scene_rate": float(np.mean([
                r["shield_intervention_count"] > 0 for r in rows
            ])),
            "s_selection_action_rate": float(sum(r["s_selection_count"] for r in rows)
                                               / max(1, sum(r["macro_action_count"] for r in rows))),
            "model_predicted_safe_but_shield_rejected_rate": float(
                sum(r["model_predicted_safe_but_shield_rejected_count"] for r in rows)
                / max(1, sum(r["model_predicted_safe_count"] for r in rows))
            ),
            "worst_candidate_B_total_minus_budget_ml": float(max(
                r["worst_candidate_B_total_minus_budget_ml"] for r in rows
            )),
            "shield_record_cache_hits": int(sum(r["shield_record_cache_hits"] for r in rows)),
            "shield_record_cache_misses": int(sum(r["shield_record_cache_misses"] for r in rows)),
            "wall_p50_p95_seconds": [float(x) for x in np.quantile(
                [r["wall_seconds"] for r in rows], [0.5, 0.95]
            )], "batch_wall_seconds": time.time() - wall0,
        }, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": result["decision"],
                      "summary": result["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
