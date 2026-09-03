"""E0 environment + contract audit (local)."""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
FROZEN = REPO / "results/clinical_window_v10_7_confirmation/frozen"
V107 = REPO / "results/clinical_window_v10_7_confirmation/shards/replication"
CHECKPOINT = REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt"
V108 = REPO / "results/clinical_window_v10_8_lazy_shield/audit"
V108.mkdir(parents=True, exist_ok=True)


def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    import numpy, torch
    env["numpy"] = numpy.__version__
    env["torch"] = torch.__version__
    env["torch_cuda_available"] = torch.cuda.is_available()

    env["git_head"] = os.popen("git rev-parse HEAD").read().strip()
    env["git_branch"] = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()
    env["git_dirty"] = os.popen("git status --short").read().strip()
    env["git_log_oneline_5"] = os.popen("git log --oneline -5").read().strip()

    (V108 / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print("[env] OK")

    inputs = {}
    files = [
        "split_replication.json", "baseline_replication.json", "splits_v10_7.json",
        "experiment_manifest.json", "sensitivity_conditions.json",
        "formal_candidate_manifest.json", "SHA256SUMS",
    ]
    sums = {}
    sums_path = FROZEN / "SHA256SUMS"
    if sums_path.exists():
        for line in sums_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            sha, fname = line.split(None, 1)
            sums[fname] = sha
    for name in files:
        p = FROZEN / name
        if p.exists():
            sha = sha256_file(p)
            inputs[str(p.relative_to(REPO))] = {
                "sha256": sha,
                "size": p.stat().st_size,
                "match_expected": (sums.get(name) == sha) if name in sums else None,
            }
    p = CHECKPOINT
    if p.exists():
        inputs[str(p.relative_to(REPO))] = {
            "sha256": sha256_file(p),
            "size": p.stat().st_size,
        }
    (V108 / "input_hashes.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False))
    print("[hashes] OK")

    shard = {}
    for c in ("C0", "C1", "C2", "C3", "C4", "C5"):
        d = V107 / c
        shard[c] = {"exists": d.exists(), "shard_count": len(list(d.glob("*.json"))) if d.exists() else 0}
    (V108 / "shard_counts.json").write_text(json.dumps(shard, indent=2, ensure_ascii=False))
    print("[shards] OK")

    eq = []
    diff = []
    for s in range(256):
        f3 = V107 / "C3" / f"clinical-d-v10.7-replication-{s:04d}.json"
        f4 = V107 / "C4" / f"clinical-d-v10.7-replication-{s:04d}.json"
        if not (f3.exists() and f4.exists()):
            continue
        d3 = json.loads(f3.read_text())
        d4 = json.loads(f4.read_text())
        if d3.get("action_sequence_hash") == d4.get("action_sequence_hash"):
            eq.append(s)
        else:
            diff.append(s)
    (V108 / "c3_vs_c4_equivalence.json").write_text(json.dumps({
        "scenes_compared": len(eq) + len(diff),
        "scenes_equal": len(eq),
        "scenes_differ": len(diff),
    }, indent=2))
    print(f"[C3 vs C4] equal on {len(eq)}/{len(eq)+len(diff)} scenes")

    # code provenance
    code_files = [
        "clinical_safety_shield_v106.py",
        "clinical_target_order_features_v106.py",
        "clinical_target_order_policy_v106.py",
        "confirmation_controllers_v107.py",
        "clinical_window_environment.py",
        "planner.py",
        "scenarios.py",
        "clinical_safety_shield_v108.py",
        "lazy_confirmation_controllers_v108.py",
        "equivalence_v108.py",
        "pilot_v108.py",
        "worst_case_v108.py",
        "smoke_lazy_shield_v108.py",
        "test_lazy_shield_v108.py",
    ]
    prov = {f: sha256_file(REPO / f) for f in code_files if (REPO / f).exists()}
    (V108 / "code_provenance.txt").write_text(
        "\n".join(f"{h}  {f}" for f, h in prov.items()) + "\n"
    )
    print("[provenance] OK")


if __name__ == "__main__":
    main()
