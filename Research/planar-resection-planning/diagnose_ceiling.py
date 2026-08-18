"""Robust ceiling estimate: scaled linear probe + small regularized MLP on the
frozen fused features, using a clean legal-only split with a standard scaler
and convergent solvers.  Confirms whether the offline classification ceiling
is really below G1 (AUROC 0.75 / bal_acc 0.70 / unsafe-FPR 0.05).
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
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score, recall_score, roc_auc_score,
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
    yt, yp = y, p
    pred = (yp >= threshold).astype(int)
    db = np.asarray([float(a.get("delta_blood", 0.0)) for a in audit])
    di = np.asarray([float(a.get("delta_ischemia", 0.0)) for a in audit])
    unsafe = (db > 0) | (di >= 0)
    return {
        "auroc": float(roc_auc_score(yt, yp)),
        "balanced_acc": float(balanced_accuracy_score(yt, pred)),
        "release_recall": float(recall_score(yt, pred, zero_division=0)),
        "unsafe_fpr": float((pred[unsafe] == 1).mean()) if unsafe.sum() else 0.0,
    }


def best_thr(y, p, audit):
    yt, yp = y, p
    best = None
    for t in np.round(np.arange(0.10, 0.95, 0.05), 2):
        bal = balanced_accuracy_score(yt, (yp >= t).astype(int))
        if best is None or bal > best[1]:
            best = (t, bal)
    th = best[0]
    return th, g1_metrics(y, p, audit, threshold=th)


def main():
    x, y, audits = load("train", limit=40_000)
    dx, dy, daudits = load("oracle_dev", limit=14_000)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    dlegal = np.asarray([bool(a.get("release_legal", True)) for a in daudits])
    xl, yl = x[legal], y[legal]
    dxl, dyl = dx[dlegal], dy[dlegal]
    dlegal_list = [a for a, ok in zip(daudits, dlegal) if ok]
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
    print("fused features...", flush=True)
    Ftr, Fva = fused_batched(model, xl), fused_batched(model, dxl)

    for tag, clf in [
        ("linear(liblinear,scaled)", make_pipeline(StandardScaler(),
             LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced"))),
        ("linear(saga,scaled)", make_pipeline(StandardScaler(),
             LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced"))),
        ("mlp(64,wd1e-3,scaled)", make_pipeline(StandardScaler(),
             MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0,
                           alpha=1e-3, early_stopping=True, validation_fraction=0.2))),
        ("mlp(128-64,wd1e-2,scaled)", make_pipeline(StandardScaler(),
             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=0,
                           alpha=1e-2, early_stopping=True, validation_fraction=0.2))),
    ]:
        clf.fit(Ftr, yl)
        ptr = clf.predict_proba(Ftr)[:, 1]
        pva = clf.predict_proba(Fva)[:, 1]
        th, met = best_thr(dyl, pva, dlegal_list)
        print(f"[{tag}] train_auroc={roc_auc_score(yl, ptr):.4f} dev_auroc={met['auroc']:.4f} "
              f"thr={th} bal_acc={met['balanced_acc']:.4f} recall={met['release_recall']:.4f} "
              f"unsafe_fpr={met['unsafe_fpr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
