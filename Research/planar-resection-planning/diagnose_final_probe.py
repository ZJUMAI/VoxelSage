"""Final decisive probe: what is the ACHIEVABLE ceiling of the offline
classification with a fully-converged linear readout on the frozen fused
features, using a large train sample?  Report G1-style metrics
(AUROC / balanced acc / release recall / unsafe-FPR / Brier).

Also compare an SGD linear readout trained with the model's own optimizer, to
confirm the readout (not the optimizer) is the key.

If the ceiling is >= ~0.75, a readout fix can reach G1; if it plateaus ~0.70,
the offline ceiling itself is below the G1 bar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    precision_score, recall_score, roc_auc_score,
)

from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy  # noqa: E402

import train_target_conditioned_clamp_oracle as m  # noqa: E402

DATA_DIR = Path("results/clinical_window_v10_2/stage1_data")
BC_MODEL = "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"


def load(name, limit=None):
    obs_parts, y_parts, audits = [], [], []
    seen = 0
    for npz in sorted(DATA_DIR.glob(f"stage1_{name}_*.npz")):
        d = np.load(npz)
        n = min(limit - seen, len(d["labels"])) if limit else len(d["labels"])
        if n <= 0:
            break
        obs_parts.append(d["obs"][:n]); y_parts.append(d["labels"][:n]); seen += n
        if limit is not None and seen >= limit:
            break
    for a in sorted(DATA_DIR.glob(f"audit_{name}_*.json")):
        audits.extend(json.loads(a.read_text(encoding="utf-8"))["audit"])
        if limit is not None and len(audits) >= seen:
            break
    return (np.concatenate(obs_parts).astype(np.float32),
            np.concatenate(y_parts).astype(np.int64), audits[: seen])


def fused_batched(model, obs, device="cpu", batch=512):
    out = []
    for start in range(0, len(obs), batch):
        x = torch.as_tensor(obs[start:start + batch], device=device)
        with torch.no_grad():
            features = model.policy.extract_features(x)
            latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, x)
        out.append(fused.detach().cpu().numpy())
    return np.concatenate(out)


def g1_metrics(y, p, audit, threshold=0.5):
    # y / p / audit are the legal-subset arrays already.
    yt, yp = y, p
    pred = (yp >= threshold).astype(int)
    db = np.asarray([float(a.get("delta_blood", 0.0)) for a in audit])
    di = np.asarray([float(a.get("delta_ischemia", 0.0)) for a in audit])
    unsafe = (db > 0) | (di >= 0)
    return {
        "auroc": float(roc_auc_score(yt, yp)),
        "auprc": float(average_precision_score(yt, yp)),
        "balanced_acc": float(balanced_accuracy_score(yt, pred)),
        "release_precision": float(precision_score(yt, pred, zero_division=0)),
        "release_recall": float(recall_score(yt, pred, zero_division=0)),
        "unsafe_fpr": float((pred[unsafe] == 1).mean()) if unsafe.sum() else 0.0,
        "brier": float(brier_score_loss(yt, yp)),
    }


def best_threshold_metrics(y, p, audit):
    yt, yp = y, p
    best = None
    for t in np.round(np.arange(0.10, 0.95, 0.05), 2):
        bal = balanced_accuracy_score(yt, (yp >= t).astype(int))
        if best is None or bal > best[1]:
            best = (t, bal)
    th, _ = best
    return th, g1_metrics(y, p, audit, threshold=th)


def main():
    x, y, audits = load("train", limit=30_000)
    dx, dy, daudits = load("oracle_dev", limit=12_000)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    dlegal = np.asarray([bool(a.get("release_legal", True)) for a in daudits])
    xl, yl = x[legal], y[legal]
    dxl, dyl = dx[dlegal], dy[dlegal]
    print(f"train legal n={len(yl)} pos={yl.sum()}  dev legal n={len(dyl)} pos={dyl.sum()}", flush=True)

    from clinical_target_conditioned_policy import TargetConditionedClampPolicy
    from tests.test_stage1_v102 import rectangle, v102_config
    bc = FrozenBCMacroTargetPolicy(BC_MODEL, device="cpu")
    clinical, reward = v102_config()
    model = m._build_frozen_base_clamp_model(
        seed=7, device="cpu", target_policy=bc, scenario=rectangle(rows=6, cols=6),
        clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    print("computing fused features (train + dev)...", flush=True)
    Ftr = fused_batched(model, xl)
    Fva = fused_batched(model, dxl)
    print(f"fused shapes: train={Ftr.shape} dev={Fva.shape}", flush=True)

    # sklearn probe (full convergence)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Ftr, yl)
    ptr = clf.predict_proba(Ftr)[:, 1]
    pva = clf.predict_proba(Fva)[:, 1]
    print(f"\n[sklearn linear probe] train_auroc={roc_auc_score(yl, ptr):.4f}", flush=True)
    dlegal_list = [a for a, ok in zip(daudits, dlegal) if ok]
    th, met = best_threshold_metrics(dyl, pva, dlegal_list)
    print(f"  dev auroc={met['auroc']:.4f} auprc={met['auprc']:.4f} threshold={th} "
          f"bal_acc={met['balanced_acc']:.4f} recall={met['release_recall']:.4f} "
          f"unsafe_fpr={met['unsafe_fpr']:.4f} brier={met['brier']:.4f}", flush=True)

    # SGD linear readout via the model's scorer (Adam, same class weight)
    print("\n[SGD linear readout via model] ...", flush=True)
    model2 = m._build_frozen_base_clamp_model(
        seed=2026090201, device="cpu", target_policy=bc, scenario=rectangle(rows=6, cols=6),
        clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    scorer = model2.policy.action_net.scorer
    model2.policy.action_net.scorer = torch.nn.Linear(scorer[0].in_features, 2)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model2.policy.action_net.scorer.weight, gain=1.0)
        model2.policy.action_net.scorer.bias.zero_()
        model2.policy.action_net.scorer.bias[1] = -4.0
    opt = torch.optim.Adam(model2.policy.action_net.scorer.parameters(), lr=1e-3, weight_decay=1e-3)
    positive = max(1, int(yl.sum())); negative = max(1, int(len(yl) - yl.sum()))
    cw = torch.as_tensor([1.0, negative / positive], dtype=torch.float32)
    rng = np.random.default_rng(2026090201)
    batch = 256
    for epoch in range(1, 11):
        idx = rng.permutation(len(yl))
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            obs = torch.as_tensor(Ftr[b], dtype=torch.float32)
            target = torch.as_tensor(yl[b], dtype=torch.long)
            logits = model2.policy.action_net.scorer(obs)
            loss = torch.nn.functional.cross_entropy(logits, target, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logits = model2.policy.action_net.scorer(torch.as_tensor(Fva, dtype=torch.float32))
            pva2 = torch.softmax(logits, 1)[:, 1].numpy()
        auroc = roc_auc_score(dyl, pva2)
        if epoch in (1, 3, 5, 10):
            print(f"  epoch {epoch:02d}: dev_auroc={auroc:.4f}", flush=True)
    th2, met2 = best_threshold_metrics(dyl, pva2, dlegal_list)
    print(f"  final auroc={met2['auroc']:.4f} threshold={th2} bal_acc={met2['balanced_acc']:.4f} "
          f"recall={met2['release_recall']:.4f} unsafe_fpr={met2['unsafe_fpr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
