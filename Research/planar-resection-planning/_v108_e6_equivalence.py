"""E6 equivalence: C4L vs C4E action hash equality on the v10.8 256 split.

Compares action_sequence_hash and other key fields per scene.  Writes
``results/clinical_window_v10_8_lazy_shield/equivalence/
e6_v108_equivalence_report.json``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
SHARDS = REPO / "results/clinical_window_v10_8_lazy_shield/shards"
V108 = REPO / "results/clinical_window_v10_8_lazy_shield/equivalence"
V108.mkdir(parents=True, exist_ok=True)


def main():
    c4e = {f.stem: json.loads(f.read_text()) for f in (SHARDS / "C4E").glob("*.json")}
    c4l = {f.stem: json.loads(f.read_text()) for f in (SHARDS / "C4L").glob("*.json")}
    sids = sorted(set(c4e) & set(c4l))
    n_equal = 0
    diffs = []
    completion_match = 0
    invariant_match = 0
    for sid in sids:
        e, l = c4e[sid], c4l[sid]
        if e.get("action_sequence_hash") == l.get("action_sequence_hash"):
            n_equal += 1
        else:
            diffs.append({"scene": sid,
                          "c4e_hash": e.get("action_sequence_hash"),
                          "c4l_hash": l.get("action_sequence_hash"),
                          "c4e_elapsed": e.get("elapsed_minutes"),
                          "c4l_elapsed": l.get("elapsed_minutes")})
        if bool(e.get("completion", False)) == bool(l.get("completion", False)):
            completion_match += 1
        if int(e.get("safety_invariant_violations", 0)) == int(l.get("safety_invariant_violations", 0)):
            invariant_match += 1
    report = {
        "scenes_compared": len(sids),
        "scenes_action_hash_equal": n_equal,
        "scenes_action_hash_differ": len(diffs),
        "scenes_completion_match": completion_match,
        "scenes_invariant_match": invariant_match,
        "first_diffs": diffs[:5],
    }
    (V108 / "e6_v108_equivalence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(f"[E6 equiv] action_hash_equal: {n_equal}/{len(sids)}")
    print(f"[E6 equiv] completion_match: {completion_match}/{len(sids)}")
    print(f"[E6 equiv] invariant_match: {invariant_match}/{len(sids)}")


if __name__ == "__main__":
    main()
