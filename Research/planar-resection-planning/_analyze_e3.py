"""Quick analysis of E3 results: verified_count distribution."""
import json
from collections import Counter
from pathlib import Path

REPO = Path("D:/26SummerCamp/VoxelSage/Research/planar-resection-planning")
equiv = REPO / "results/clinical_window_v10_8_lazy_shield/equivalence"

verified_count_dist = Counter()
verified_mean_sum = 0.0
verified_mean_n = 0
report = json.loads((equiv / "equivalence_report.json").read_text())
for r in report:
    if r.get("missing_v107_c4_ref"):
        continue
    # We don't have per-step counts; approximate by max
    verified_count_dist[r["c4l_verified_max"]] += 1
    verified_mean_sum += r["c4l_verified_mean"]
    verified_mean_n += 1

print(f"Scenes analysed: {verified_mean_n}")
print(f"verified_count_max distribution: {dict(sorted(verified_count_dist.items()))}")
print(f"verified_count_mean (per-scene average): {verified_mean_sum/verified_mean_n:.3f}")

# Detailed per-step distribution (requires per-step records; we have summary only)
# So we also look at the JSON shards
n_steps_total = 0
verified_steps = Counter()
import json as _j
for shard_path in (equiv / "C4L").glob("*.json"):
    s = _j.loads(shard_path.read_text())
    n_steps_total += s["macro_action_count"]
    for _ in range(s["macro_action_count"]):
        # Each step verified_count is in the selected_rank_distribution; we don't have per-step
        # but verified_count_mean is per-step
        pass

# Print the summary
summary = json.loads((equiv / "C4L_summary.json").read_text())
print()
print(json.dumps(summary, indent=2))
