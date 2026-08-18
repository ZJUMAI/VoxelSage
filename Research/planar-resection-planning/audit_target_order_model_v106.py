"""Offline teacher-ranking and tail-risk audit for a v10.6 checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path: sys.path.insert(0, str(SIM))

from clinical_target_order_policy_v106 import TargetOrderScorerV106, unpack_spatial
from train_target_order_v106 import best_safe_indices, normalize_candidate_batch

BASE = SIM / "results/clinical_window_v10_6_shielded_learning"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=BASE / "teacher/teacher_rankings_v106.npz")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    try: checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError: checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TargetOrderScorerV106(
        hidden=int(checkpoint["hidden"]), spatial=int(checkpoint["spatial"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    archive = np.load(args.teacher)
    data = {key: archive[key] for key in archive.files}; archive.close()
    n = len(data["global_context"])
    best = best_safe_indices(data["T_total"], data["B_total"],
                             data["safe_exact"], data["valid"])
    predictions = {name: np.full_like(data["T_total"], np.nan, dtype=np.float32)
                   for name in ("score", "T_total", "B_tail", "B_total",
                                "completion_logit", "safe_logit")}
    for start in range(0, n, args.batch_size):
        idx = np.arange(start, min(start + args.batch_size, n))
        grid = torch.from_numpy(unpack_spatial(data["grid_bits"][idx], data["transfer_q"][idx])).to(device)
        feature = normalize_candidate_batch(data["features"][idx], checkpoint["feature_scales"])
        with torch.no_grad():
            out = model(
                grid, torch.from_numpy(feature).to(device),
                torch.from_numpy(data["global_context"][idx].astype(np.float32)).to(device),
                torch.from_numpy(data["targets"][idx].astype(np.int64)).to(device),
            )
        for name in predictions: predictions[name][idx] = out[name].cpu().numpy()
    predictions["T_total"] *= float(checkpoint["time_scale"])
    predictions["B_tail"] *= float(checkpoint["blood_scale"])
    predictions["B_total"] *= float(checkpoint["blood_scale"])
    top1 = top3 = 0; ndcg = []
    for i in range(n):
        safe = np.flatnonzero(data["safe_exact"][i] & data["valid"][i])
        order = safe[np.argsort(-predictions["score"][i, safe])]
        top1 += int(order[0] == best[i]); top3 += int(best[i] in order[:3])
        truth = sorted(safe, key=lambda j: (round(float(data["T_total"][i, j]), 6),
                                            round(float(data["B_total"][i, j]), 6), int(j)))
        relevance = {candidate: 1.0 / (rank + 1) for rank, candidate in enumerate(truth)}
        k = min(3, len(order))
        dcg = sum(relevance[int(order[r])] / np.log2(r + 2) for r in range(k))
        idcg = sum((1.0 / (r + 1)) / np.log2(r + 2) for r in range(k))
        ndcg.append(dcg / idcg if idcg else 0.0)
    valid = data["valid"]
    unsafe = valid & ~data["safe_exact"]
    b_tail_error = predictions["B_tail"][valid] - data["B_tail"][valid]
    b_total_error = predictions["B_total"][valid] - data["B_total"][valid]
    safe_pred = torch.sigmoid(torch.from_numpy(predictions["safe_logit"])).numpy() >= 0.5
    result = {
        "version": "v10.6-offline-model-audit-v1", "checkpoint": str(args.checkpoint),
        "n_states": n, "n_candidates": int(valid.sum()),
        "safe_set_top1": top1 / n, "safe_set_top3": top3 / n,
        "safe_set_ndcg_at_3": float(np.mean(ndcg)),
        "B_tail_mae_ml": float(np.mean(np.abs(b_tail_error))),
        "B_tail_underestimate_p95_max_ml": [
            float(np.quantile(np.maximum(-b_tail_error, 0), 0.95)),
            float(np.max(np.maximum(-b_tail_error, 0))),
        ],
        "B_total_mae_ml": float(np.mean(np.abs(b_total_error))),
        "B_total_underestimate_p95_max_ml": [
            float(np.quantile(np.maximum(-b_total_error, 0), 0.95)),
            float(np.max(np.maximum(-b_total_error, 0))),
        ],
        "unsafe_false_negative_rate": float(safe_pred[unsafe].mean()) if unsafe.any() else 0.0,
        "completion_recall": float((predictions["completion_logit"][data["completion"] & valid] >= 0).mean()),
        "note": "Risk predictions are diagnostic only; the exact external shield grants execution.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
