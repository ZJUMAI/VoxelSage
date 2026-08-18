"""Overfit test: can the Stage-1 model rank the TRAIN labels at all?

If train AUROC plateaus ~0.55-0.6 even with 15+ epochs and a higher LR, the
label is not recoverable from the observation (structural).  If train AUROC
climbs toward 1.0 while dev stays ~0.5, the mapping is learnable and the
problem is generalization / data distribution.

Ablations:
  * full (plan_spatial trainable)  vs  scorer-only (plan_spatial frozen): is
    the trainable plan branch helping at all?
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy  # noqa: E402

import train_target_conditioned_clamp_oracle as m  # noqa: E402

DATA_DIR = Path("results/clinical_window_v10_2/stage1_data")
BC_MODEL = "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"


def load_split(name, limit=None):
    obs_parts, y_parts, reg_parts, audits = [], [], [], []
    seen = 0
    for npz in sorted(DATA_DIR.glob(f"stage1_{name}_*.npz")):
        d = np.load(npz)
        n = len(d["labels"]) if limit is None else min(limit - seen, len(d["labels"]))
        if n <= 0:
            break
        obs_parts.append(d["obs"][:n])
        y_parts.append(d["labels"][:n])
        reg_parts.append(d["regression"][:n])
        seen += n
        if limit is not None and seen >= limit:
            break
    for a in sorted(DATA_DIR.glob(f"audit_{name}_*.json")):
        audits.extend(json.loads(a.read_text(encoding="utf-8"))["audit"])
        if limit is not None and len(audits) >= seen:
            break
    return (
        np.concatenate(obs_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int64),
        np.concatenate(reg_parts).astype(np.float32),
        audits[: seen],
    )


def run(freeze_plan: bool, lr: float, epochs: int, n_train: int, device="cpu"):
    x, y, reg, audit = load_split("train", limit=n_train)
    dx, dy, dreg, daudit = load_split("oracle_dev", limit=20_000)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audit])
    dlegal = np.asarray([bool(a.get("release_legal", True)) for a in daudit])
    print(f"\n=== freeze_plan={freeze_plan} lr={lr} epochs={epochs} n_train={len(y)} ===")
    print(f"train legal={legal.sum()} pos={int(y[legal].sum())}  dev legal={dlegal.sum()}")

    from clinical_target_conditioned_policy import TargetConditionedClampPolicy
    from tests.test_stage1_v102 import rectangle, v102_config
    bc = FrozenBCMacroTargetPolicy(BC_MODEL, device=device)
    clinical, reward = v102_config()
    model = m._build_frozen_base_clamp_model(
        seed=2026090201, device=device, target_policy=bc,
        scenario=rectangle(rows=6, cols=6),
        clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    if freeze_plan:
        for p in model.policy.features_extractor.plan_spatial.parameters():
            p.requires_grad_(False)
    params = list(model.policy.action_net.parameters())
    if not freeze_plan:
        params += list(model.policy.features_extractor.plan_spatial.parameters())
    params += list(model.policy.regression_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    positive = max(1, int(y.sum()))
    negative = max(1, int(len(y) - y.sum()))
    cw = torch.as_tensor([1.0, negative / positive], dtype=torch.float32, device=device)
    rng = np.random.default_rng(2026090201)
    batch = 256
    best_train_auroc = -1.0
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(y))
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            obs = torch.as_tensor(x[b], device=device)
            target = torch.as_tensor(y[b], dtype=torch.long, device=device)
            dt = torch.as_tensor(reg[b], dtype=torch.float32, device=device)
            features = model.policy.extract_features(obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, obs)
            logits = model.policy.action_net.scorer(fused)
            reg_out = model.policy.regression_head(fused)
            cls = torch.nn.functional.cross_entropy(logits, target, weight=cw)
            rl = torch.nn.functional.mse_loss(reg_out, dt)
            loss = cls + 0.1 * rl
            opt.zero_grad()
            loss.backward()
            opt.step()
        trp = m._release_probabilities(model, x[legal], device)
        tr_auroc = roc_auc_score(y[legal], trp)
        dp = m._release_probabilities(model, dx[dlegal], device)
        dv_auroc = roc_auc_score(dy[dlegal], dp)
        best_train_auroc = max(best_train_auroc, tr_auroc)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  epoch {epoch:02d}: train_auroc={tr_auroc:.4f} dev_auroc={dv_auroc:.4f}")
    print(f"  BEST train_auroc={best_train_auroc:.4f}")


if __name__ == "__main__":
    # 1) scorer-only (plan frozen), higher LR, 12 epochs
    run(freeze_plan=True, lr=3e-4, epochs=12, n_train=12_000)
    # 2) full (plan trainable)
    run(freeze_plan=False, lr=3e-4, epochs=12, n_train=12_000)
