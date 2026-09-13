"""Validate and summarize the corrected v10.8 sensitivity run, entirely offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
SOURCE = V108 / "sensitivity_condition_budget_20260913"
OUTPUT = REPO / "artifacts/v10.8-condition-sensitivity"
CONTROLLERS = ("C0", "C4L", "C5")
CONDITIONS = {
    "S0": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S1": {"max_clamp_minutes": 12.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S2": {"max_clamp_minutes": 10.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S3": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.5},
    "S4": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.25},
}
BUDGET_SEMANTICS = "same_condition_c0_plus_frozen_margin_v1"
EPS = 1e-9


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    require(path.is_file(), f"missing required input: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # The original frozen split was written with the Windows code page.
        # Its original bytes are still verified against the manifest hash.
        text = raw.decode("gb18030")
    return json.loads(text)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stats(values):
    if not values:
        return {"n": 0}
    values = sorted(float(x) for x in values)
    return {
        "n": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "min": values[0], "max": values[-1],
        "upper_cvar10": statistics.mean(values[-max(1, math.ceil(0.1 * len(values))):]),
    }


def complete(row):
    return row["completion"] and row["failure_reason"] is None


def paired(rows, reference, ids):
    # A failed early stop does not represent a completed-plan time saving.
    usable = [sid for sid in ids if complete(rows[sid]) and complete(reference[sid])]
    return {
        "n_paired_complete": len(usable), "n_excluded_failure": len(ids) - len(usable),
        "delta_T_min": stats([rows[sid]["elapsed_minutes"] - reference[sid]["elapsed_minutes"] for sid in usable]),
        "delta_B_ml": stats([rows[sid]["realized_episode_B_ml"] - reference[sid]["realized_episode_B_ml"] for sid in usable]),
    }


def aggregate(source, split_file, reference_file, checkpoint, runner, expected_count=128):
    manifest_path = source / "run_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("budget_semantics") == BUDGET_SEMANTICS, "wrong budget semantics")
    require(manifest.get("semantics") == "lazy_exact_fail_closed_no_fallback", "wrong C4L semantics")
    require(manifest.get("conditions") == list(CONDITIONS), "expected exactly S0--S4 in manifest")
    require(manifest.get("controllers") == list(CONTROLLERS), "expected exactly C0/C4L/C5 in manifest")
    require(manifest.get("limit") == expected_count, "manifest count differs from requested audit count")
    require(manifest.get("runner") == runner.name, "runner filename mismatch")
    require(Path(manifest["output_root"]).resolve() == source.resolve(), "manifest output root mismatch")
    input_paths = {
        "split_sha256": split_file, "reference_s0_baseline_sha256": reference_file,
        "checkpoint_sha256": checkpoint, "runner_sha256": runner,
    }
    for key, path in input_paths.items():
        require(path.is_file(), f"missing provenance input: {path}")
        require(manifest.get(key) == sha256(path), f"input hash mismatch: {path}")
    require(isinstance(manifest.get("repository_commit"), str) and len(manifest["repository_commit"]) == 40,
            "missing repository commit")
    scenes = read_json(split_file)["scenarios"]
    offset = manifest["offset"]
    require(isinstance(offset, int) and offset >= 0, "invalid scenario offset")
    ids = [scene["scenario_id"] for scene in scenes[offset:offset + expected_count]]
    require(len(ids) == expected_count and len(set(ids)) == expected_count, "wrong or duplicate scenario subset")
    margin = float(manifest["margin_ml"])
    require(math.isfinite(margin) and margin >= 0, "invalid fixed margin")
    metadata = {key: value for key, value in manifest.items() if key not in {
        "controllers", "conditions", "offset", "limit", "scene_workers", "output_root",
    }}
    result = {}
    for condition, cfg in CONDITIONS.items():
        baseline_path = source / f"baseline_{condition}.json"
        baseline = read_json(baseline_path)
        expected_config = {"early_end_mode": "disabled", "early_end_minutes": 0.0,
                           "max_steps_multiplier": 8.0, **cfg}
        require(baseline.get("budget_semantics") == BUDGET_SEMANTICS, f"wrong baseline semantics: {condition}")
        require(baseline.get("condition") == condition, f"wrong baseline condition: {condition}")
        require(baseline.get("condition_config") == expected_config, f"wrong baseline config: {condition}")
        require(baseline.get("split_sha256") == manifest["split_sha256"], f"wrong baseline split: {condition}")
        require(baseline.get("scenario_ids") == ids and set(baseline["records"]) == set(ids),
                f"baseline scenario mismatch: {condition}")
        baseline_hash = sha256(baseline_path)
        expected_metadata = {**metadata, "condition": condition, "condition_config": cfg,
                             "baseline_sha256": baseline_hash}
        controllers = {}
        summaries = {}
        shard_hashes = {}
        for controller in CONTROLLERS:
            directory = source / condition / controller
            paths = list(directory.glob("*.json"))
            require({p.stem for p in paths} == set(ids),
                    f"incomplete or extra shards: {condition}/{controller}: {len(paths)}/{expected_count}")
            rows = {}
            hashes = {}
            for sid in ids:
                path = directory / f"{sid}.json"
                row = read_json(path)
                require((row.get("condition"), row.get("controller"), row.get("scenario_id")) ==
                        (condition, controller, sid), f"shard identity mismatch: {path}")
                require(row.get("evaluation_metadata") == expected_metadata, f"shard provenance mismatch: {path}")
                require(isinstance(row["completion"], bool), f"invalid completion flag: {path}")
                base = baseline["records"][sid]
                require(base.get("scenario_id") == sid and base.get("controller") == "C0" and complete(base),
                        f"invalid C0 baseline: {condition}/{sid}")
                for key in ("elapsed_minutes", "realized_episode_B_ml", "budget_ml"):
                    require(math.isfinite(row[key]) and row[key] >= 0, f"invalid {key}: {path}")
                require(abs(row["budget_ml"] - base["realized_episode_B_ml"] - margin) <= EPS,
                        f"wrong same-condition budget: {path}")
                if controller == "C0":
                    for key in ("elapsed_minutes", "realized_episode_B_ml", "action_sequence_hash"):
                        require(row[key] == base[key], f"C0 differs from frozen baseline ({key}): {path}")
                rows[sid] = row
                hashes[sid] = sha256(path)
            controllers[controller] = rows
            shard_hashes[controller] = content_hash(hashes)
            summaries[controller] = {
                "n": len(rows), "completes": sum(complete(r) for r in rows.values()),
                "failures": sum(not complete(r) for r in rows.values()),
                "infeasibles": sum(r["failure_reason"] == "infeasible_no_safe_candidate" for r in rows.values()),
                # Actual blood loss is checked even if an episode later aborts.
                "overruns": sum(r["realized_episode_B_ml"] > r["budget_ml"] + EPS for r in rows.values()),
                "failure_and_overrun": sum(not complete(r) and r["realized_episode_B_ml"] > r["budget_ml"] + EPS
                                           for r in rows.values()),
                "invariant_events": sum(r["safety_invariant_violations"] for r in rows.values()),
                "episodes_with_invariant_violation": sum(r["safety_invariant_violations"] > 0 for r in rows.values()),
                "observed_T_min_all_episodes": stats([r["elapsed_minutes"] for r in rows.values()]),
                "observed_B_ml_all_episodes": stats([r["realized_episode_B_ml"] for r in rows.values()]),
                "realized_B_minus_budget_ml": stats([r["realized_episode_B_ml"] - r["budget_ml"] for r in rows.values()]),
            }
        baseline_mean = summaries["C0"]["observed_B_ml_all_episodes"]["mean"]
        result[condition] = {
            "condition_config": cfg, "baseline_sha256": baseline_hash,
            "controller_shard_set_sha256": shard_hashes, "controllers": summaries,
            "margin_divided_by_mean_C0_B": margin / baseline_mean if baseline_mean else None,
            "C4L_minus_C0": paired(controllers["C4L"], controllers["C0"], ids),
            "C5_minus_C0": paired(controllers["C5"], controllers["C0"], ids),
            "C4L_minus_C5": paired(controllers["C4L"], controllers["C5"], ids),
        }
    return {
        "version": "v10.8-condition-budget-sensitivity-v1", "n_per_condition_controller": expected_count,
        "verified_shards": expected_count * len(CONDITIONS) * len(CONTROLLERS),
        "source_root": str(source.resolve()), "source_manifest": manifest,
        "source_manifest_sha256": sha256(manifest_path),
        "input_files": {key: str(path.resolve()) for key, path in input_paths.items()},
        "report_script_sha256": sha256(Path(__file__)), "scenario_ids": ids, "margin_ml": margin,
        "root_cause": "The old v10.8 runners reused S0 C0 blood loss for every condition; S1/S2 were often over-restricted and S3/S4 often over-permissive. Each corrected budget uses direct C0 under that condition plus the unchanged fixed margin.",
        "definitions": {
            "complete": "completion=true and failure_reason=null; failure is the complement",
            "overrun": "realized_episode_B_ml > budget_ml + 1e-9, independently of completion/infeasibility",
            "paired_deltas": "named first controller minus second, same scenarios where both completed; failures explicitly excluded",
            "upper_cvar10": "mean of the largest ceil(10% * n) values; descriptive, no significance test",
            "shard_set_sha256": "SHA256 of sorted-key compact JSON mapping scenario_id to each shard file SHA256",
        },
        "conditions": result,
    }


def render_report(summary):
    lines = [
        "# v10.8 同条件预算敏感性纠正", "",
        "原 v10.8 入口把 S0 的蛇形出血量用于全部条件，导致 S1/S2 的部分场景预算过紧、S3/S4 的部分场景预算过宽。此次按各条件重算 C0，再加同一个固定裕量；不修改模型、候选、安全判据或旧实验记录。", "",
        f"已核对 {summary['verified_shards']} 个分片，每个条件、控制器 {summary['n_per_condition_controller']} 个相同场景。固定裕量 {summary['margin_ml']:.12f} mL。", "",
        "完成与实际超预算分别统计；失败或不可行的场景也会检查已经发生的血量是否超预算。配对时间和血量只比较双方均完成的场景。以下是描述统计，不作显著性或临床有效性判断。", "",
        "| 条件 | 控制器 | 完成 | 失败 | 不可行 | 实际超预算 | 安全不变量事件 |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition, item in summary["conditions"].items():
        for controller, row in item["controllers"].items():
            lines.append(f"| {condition} | {controller} | {row['completes']} | {row['failures']} | {row['infeasibles']} | {row['overruns']} | {row['invariant_events']} |")
    lines += ["", "| 条件 | 配对差值方向 | 配对完成数 | 排除失败数 | 平均 ΔT (min) | 平均 ΔB (mL) |", "|---|---|---:|---:|---:|---:|"]
    for condition, item in summary["conditions"].items():
        for contrast in ("C4L_minus_C0", "C5_minus_C0", "C4L_minus_C5"):
            row = item[contrast]
            dt = f"{row['delta_T_min']['mean']:.6f}" if row["n_paired_complete"] else "NA"
            db = f"{row['delta_B_ml']['mean']:.6f}" if row["n_paired_complete"] else "NA"
            lines.append(f"| {condition} | {contrast} | {row['n_paired_complete']} | {row['n_excluded_failure']} | {dt} | {db} |")
    lines += ["", "来源与复现：", "", f"- 源目录：`{summary['source_root']}`",
              f"- Manifest SHA256：`{summary['source_manifest_sha256']}`",
              f"- Runner SHA256：`{summary['source_manifest']['runner_sha256']}`",
              f"- Checkpoint SHA256：`{summary['source_manifest']['checkpoint_sha256']}`",
              f"- 汇总脚本 SHA256：`{summary['report_script_sha256']}`",
              "- 每个条件的基线哈希、各控制器分片集合哈希、场景清单、配对分布与裕量/平均基线血量比见 `summary.json`。", "",
              "评估命令须使用已安装 PyTorch 的研究环境；汇总脚本只需 Python 标准库。", "",
              "```text", summary["evaluation_command"], summary["report_command"], "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split-file", type=Path, default=V108 / "frozen/split_lazy_replication.json")
    parser.add_argument("--reference-baseline-file", type=Path, default=V108 / "frozen/baseline_lazy_replication.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--expected-count", type=int, default=128, help="128 for the paper; smaller values only for explicit smoke audits")
    parser.add_argument("--evaluation-python", default="python", help="Interpreter for the recorded rerun command; must contain PyTorch")
    args = parser.parse_args(argv)
    require(args.expected_count > 0, "expected count must be positive")
    runner = REPO / "evaluate_v108_sensitivity.py"
    summary = aggregate(args.source, args.split_file, args.reference_baseline_file, args.checkpoint, runner, args.expected_count)
    manifest = summary["source_manifest"]
    summary["evaluation_command"] = subprocess.list2cmdline([
        args.evaluation_python, str(runner), "--controllers", ",".join(CONTROLLERS), "--conditions", ",".join(CONDITIONS),
        "--limit", str(args.expected_count), "--offset", str(manifest["offset"]),
        "--scene-workers", str(manifest["scene_workers"]), "--margin", str(manifest["margin_ml"]),
        "--split-file", str(args.split_file), "--baseline-file", str(args.reference_baseline_file),
        "--checkpoint", str(args.checkpoint), "--output-root", str(args.source),
    ])
    summary["report_command"] = subprocess.list2cmdline([
        sys.executable, str(Path(__file__)), "--source", str(args.source), "--output", str(args.output),
        "--split-file", str(args.split_file), "--reference-baseline-file", str(args.reference_baseline_file),
        "--checkpoint", str(args.checkpoint), "--expected-count", str(args.expected_count),
        "--evaluation-python", args.evaluation_python,
    ])
    # Do not create or replace any report until all source checks pass.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Verified {summary['verified_shards']} shards; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
