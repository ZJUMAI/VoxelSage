"""Spatial-vs-pooled readout probe: is the label reachable from the frozen
base features IF the scorer can see spatial layout?

Probes on train legal samples:
  1. current 173-dim fused features (reproduce ~0.62)
  2. base_spatial feature map region-pooled into a 5x5 grid  (32*25 = 800 dims)
  3. raw base obs channels region-pooled into a 5x5 grid    (26*25 = 650 dims)

If (2)/(3) jump well above 0.62, the bottleneck is the pooled fused_features
representation, and exposing spatial layout would fix Stage 1 (architecture
change -> decision needed).  If all plateau ~0.6, the frozen features lack the
signal entirely.
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
from sklearn.metrics import roc_auc_score  # noqa: E402

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
        obs_parts.append(d["obs"][:n])
        y_parts.append(d["labels"][:n])
        seen += n
        if limit is not None and seen >= limit:
            break
    for a in sorted(DATA_DIR.glob(f"audit_{name}_*.json")):
        audits.extend(json.loads(a.read_text(encoding="utf-8"))["audit"])
        if limit is not None and len(audits) >= seen:
            break
    return (np.concatenate(obs_parts).astype(np.float32),
            np.concatenate(y_parts).astype(np.int64), audits[: seen])


def region_pool(feat, grid=5):
    # feat: (B, C, H, W) -> (B, C*grid*grid) mean over each grid cell
    B, C, H, W = feat.shape
    gh, gw = H // grid, W // grid
    out = np.zeros((B, C, grid, grid), dtype=feat.dtype)
    for i in range(grid):
        for j in range(grid):
            out[:, :, i, j] = feat[:, :, i*gh:(i+1)*gh, j*gw:(j+1)*gw].mean(axis=(2, 3))
    return out.reshape(B, -1)


def base_spatial_map(model, obs, device="cpu"):
    x = torch.as_tensor(obs, device=device)
    with torch.no_grad():
        return model.policy.features_extractor.base_spatial(x[:, :26]).detach().cpu().numpy()


def probe(Xtr, ytr, Xva, yva, tag):
    clf = LogisticRegression(max_iter=800, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    print(f"{tag}: train_auroc={roc_auc_score(ytr, clf.predict_proba(Xtr)[:,1]):.4f} "
          f"dev_auroc={roc_auc_score(yva, p):.4f}")


def main():
    x, y, audits = load("train", limit=16_000)
    dx, dy, daudits = load("oracle_dev", limit=20_000)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    dlegal = np.asarray([bool(a.get("release_legal", True)) for a in daudits])
    xl, yl = x[legal], y[legal]
    dxl, dyl = dx[dlegal], dy[dlegal]
    rng = np.random.default_rng(0)
    tr_take = np.sort(rng.choice(len(yl), size=min(6000, len(yl)), replace=False))
    xl, yl = xl[tr_take], yl[tr_take]
    print(f"train legal n={len(yl)} pos={yl.sum()}  dev legal n={len(dyl)}")

    from clinical_target_conditioned_policy import TargetConditionedClampPolicy
    from tests.test_stage1_v102 import rectangle, v102_config
    bc = FrozenBCMacroTargetPolicy(BC_MODEL, device="cpu")
    clinical, reward = v102_config()
    model = m._build_frozen_base_clamp_model(
        seed=7, device="cpu", target_policy=bc, scenario=rectangle(rows=6, cols=6),
        clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    # 1) current fused 173-dim
    fused_tr = fused_features(model, xl)
    fused_va = fused_features(model, dxl)
    probe(fused_tr, yl, fused_va, dyl, "fused173")
    # 2) region-pooled base_spatial features
    bmap_tr = base_spatial_map(model, xl).astype(np.float32)
    bmap_va = base_spatial_map(model, dxl).astype(np.float32)
    probe(region_pool(bmap_tr), yl, region_pool(bmap_va), dyl, "base_spatial_region5x5")
    # 3) raw base channels region-pooled
    probe(region_pool(xl[:, :26]), yl, region_pool(dxl[:, :26]), dyl, "raw_base_ch_region5x5")
    # 4) raw base channels + full-plan channels (26,27) flattened coarse (10x13 grid)
    probe(np.concatenate([region_pool(xl[:, :26], 10), region_pool(xl[:, 26:28], 10)], 1),
          yl,
          np.concatenate([region_pool(dxl[:, :26], 10), region_pool(dxl[:, 26:28], 10)], 1),
          dyl, "raw_base+plan_region10x13")


def fused_features(model, obs, device="cpu"):
    x = torch.as_tensor(obs, device=device)
    with torch.no_grad():
        features = model.policy.extract_features(x)
        latent_pi, _ = model.policy.mlp_extractor(features)
        fused = model.policy.action_net.fused_features(latent_pi, x)
    return fused.detach().cpu().numpy()


if __name__ == "__main__":
    main()
