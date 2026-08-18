"""Gate B behavior cloning: pairwise ranking BC over the teacher candidates.

Reads the teacher ranking data (guide 7.4: full candidate ranking / cost
difference, not just top-1), trains ``TargetOrderScorer`` with a pairwise
ranking (hinge) loss, reports top-1 / top-3 / NDCG against the teacher's frozen
branch rule, and saves a checkpoint for the deterministic rollout admission.

Teacher preference order (guide 6.2 / 7.4), per decision state:
    completion (True first) -> blood-safe (cost_B <= threshold) -> shortest
    total time -> less blood.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_target_order_features import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    normalize_features,
)
from clinical_target_order_policy import TargetOrderScorer  # noqa: E402

TEACHER_DIR = SIM / "results/clinical_window_v10_4_target_order/teacher"
RUNS_DIR = SIM / "results/clinical_window_v10_4_target_order/runs"
PAIR_MARGIN = 1.0
PAIRS_PER_STATE = 16


def teacher_order(cost_T, cost_B, comp, safe):
    """Teacher preference order indices (best first)."""
    # keys: (not completion, not safe, cost_T, cost_B)
    keys = [(-int(c), -int(s), float(t), float(b))
            for c, s, t, b in zip(comp, safe, cost_T, cost_B)]
    return sorted(range(len(keys)), key=lambda i: keys[i])


def build_pairs(cost_T, cost_B, comp, safe):
    """Return list of (i, j) with i strictly preferred over j."""
    order = teacher_order(cost_T, cost_B, comp, safe)
    pairs = []
    for ai, i in enumerate(order):
        for j in order[ai + 1:]:
            pairs.append((i, j))
            if len(pairs) >= PAIRS_PER_STATE:
                return pairs
    return pairs


def ndcg_at_k(model, feat, glob, cost_T, cost_B, comp, safe, k=3):
    """Mean NDCG@k over states using teacher relevance = order-derived."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        B = feat.shape[0]
        for si in range(B):
            valid = ~np.isnan(cost_T[si])
            f = torch.from_numpy(feat[si, valid]).unsqueeze(0)
            g = torch.from_numpy(glob[si]).unsqueeze(0)
            logits = model(f, g).squeeze(0).cpu().numpy()
            idx = np.flatnonzero(valid)
            ct, cb, c, s = cost_T[si, idx], cost_B[si, idx], comp[si, idx], safe[si, idx]
            order = teacher_order(ct, cb, c, s)
            rel = np.zeros(len(order))
            for r, cand in enumerate(order):
                rel[cand] = 1.0 / (r + 1)  # relevance by teacher rank
            pred_order = np.argsort(-logits)
            dcg = sum(rel[pred_order[qi]] / np.log2(qi + 2) for qi in range(min(k, len(pred_order))))
            idcg = sum(1.0 / (r + 1) / np.log2(r + 2) for r in range(min(k, len(order))))
            total += dcg / idcg if idcg > 0 else 0.0
            n += 1
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-npz", type=Path, default=TEACHER_DIR / "teacher_rankings.npz")
    parser.add_argument("--scales", type=Path, default=TEACHER_DIR / "feature_scales.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-states", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "target_order_bc.pt")
    parser.add_argument("--limit-states", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.teacher_npz)
    feat = d["features"]; glob = d["global_context"]
    cost_T = d["cost_T"]; cost_B = d["cost_B"]
    comp = d["completion"]; safe_thr = d["safe_threshold"]
    if args.limit_states:
        feat = feat[:args.limit_states]; glob = glob[:args.limit_states]
        cost_T = cost_T[:args.limit_states]; cost_B = cost_B[:args.limit_states]
        comp = comp[:args.limit_states]; safe_thr = safe_thr[:args.limit_states]

    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    N, K, Dc = feat.shape
    Dg = glob.shape[1]
    print(f"states={N} max_k={K} cand_dim={Dc} global_dim={Dg}", flush=True)

    # Normalise geometric features with Train-only scales.
    norm = np.zeros_like(feat, dtype=np.float32)
    for si in range(N):
        for ki in range(K):
            if not np.isnan(cost_T[si, ki]):
                norm[si, ki] = normalize_features(feat[si, ki], scales)
    valid_mask = ~np.isnan(cost_T)
    # cost_B <= threshold broadcasts (N,K) vs (N,1); NaN cost_B compares False,
    # so invalid candidates are naturally excluded and valid_mask stays consistent.
    safe = (cost_B <= safe_thr[:, None]) & valid_mask

    model = TargetOrderScorer(cand_dim=Dc, global_dim=Dg, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params}", flush=True)

    g_t = torch.from_numpy(glob)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        order_idx = np.arange(N)
        np.random.shuffle(order_idx)
        for start in range(0, N, args.batch_states):
            batch = order_idx[start:start + args.batch_states]
            # Gather pairwise comparisons across the batch.
            pair_i, pair_j, batch_s = [], [], []
            for si in batch:
                valid = ~np.isnan(cost_T[si])
                idx = np.flatnonzero(valid)
                if len(idx) < 2:
                    continue
                pairs = build_pairs(
                    cost_T[si, idx], cost_B[si, idx], comp[si, idx], safe[si, idx])
                for (a, b) in pairs:
                    pair_i.append(idx[a]); pair_j.append(idx[b]); batch_s.append(si)
            if not pair_i:
                continue
            # Pair slices are (P, Dc); reshape to (P, 1, Dc) for the shared scorer.
            b_feat = torch.from_numpy(norm[batch_s, pair_i]).unsqueeze(1)
            j_feat = torch.from_numpy(norm[batch_s, pair_j]).unsqueeze(1)
            b_g = g_t[batch_s]
            l_i = model(b_feat, b_g).squeeze(1)
            l_j = model(j_feat, b_g).squeeze(1)
            loss = torch.clamp(PAIR_MARGIN - (l_i - l_j), min=0.0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        print(f"epoch {epoch + 1}/{args.epochs}: ranking_loss={total_loss / max(1, n_batches):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Ranking quality on the training data.
    model.eval()
    top1 = top3 = n = 0
    with torch.no_grad():
        for si in range(N):
            valid = ~np.isnan(cost_T[si])
            idx = np.flatnonzero(valid)
            if len(idx) < 2:
                continue
            f = torch.from_numpy(norm[si, idx]).unsqueeze(0)
            g = torch.from_numpy(glob[si]).unsqueeze(0)
            logits = model(f, g).squeeze(0).cpu().numpy()
            ct, cb, c, s = cost_T[si, idx], cost_B[si, idx], comp[si, idx], safe[si, idx]
            order = teacher_order(ct, cb, c, s)
            best = order[0]
            pred_order = np.argsort(-logits)
            top1 += int(pred_order[0] == best)
            top3 += int(best in pred_order[:3])
            n += 1
    ndcg = ndcg_at_k(model, norm, glob, cost_T, cost_B, comp, safe)
    print(f"teacher-learning: top1={top1 / max(1, n):.4f} top3={top3 / max(1, n):.4f} "
          f"ndcg@3={ndcg:.4f} (n={n})", flush=True)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    meta = {
        "version": "v10.4-target-order-bc-v1",
        "top1_acc": top1 / max(1, n),
        "top3_acc": top3 / max(1, n),
        "ndcg_at_3": ndcg,
        "n_states": N,
        "checkpoint": str(args.checkpoint),
    }
    (args.checkpoint.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
