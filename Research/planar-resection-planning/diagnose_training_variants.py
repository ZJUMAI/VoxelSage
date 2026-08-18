"""Decisive training-variant experiment: why does CNN training reach only 0.53
dev AUROC when a linear probe on the SAME fused features reaches 0.70?

Variants (all use the same frozen-base model + existing fused features):
  V1  all samples, class_weight=[1, neg/pos]          (reproduce ~0.53)
  V2  legal-only samples, balanced class weight       (mirror the probe)
  V3  all samples, legal up-weighted (legal=1, nonlegal=0.1)
  V4  legal-only + scorer re-init with gain=1.0 (healthy pre-activations)

Report dev AUROC (legal subset) after each epoch on a fast subsample.
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


def load(name, limit=None):
    obs_parts, y_parts, reg_parts, audits = [], [], [], []
    seen = 0
    for npz in sorted(DATA_DIR.glob(f"stage1_{name}_*.npz")):
        d = np.load(npz)
        n = min(limit - seen, len(d["labels"])) if limit else len(d["labels"])
        if n <= 0:
            break
        obs_parts.append(d["obs"][:n]); y_parts.append(d["labels"][:n])
        reg_parts.append(d["regression"][:n]); seen += n
        if limit is not None and seen >= limit:
            break
    for a in sorted(DATA_DIR.glob(f"audit_{name}_*.json")):
        audits.extend(json.loads(a.read_text(encoding="utf-8"))["audit"])
        if limit is not None and len(audits) >= seen:
            break
    return (np.concatenate(obs_parts).astype(np.float32),
            np.concatenate(y_parts).astype(np.int64),
            np.concatenate(reg_parts).astype(np.float32), audits[: seen])


def run_variant(name, *, legal_only, nonlegal_weight, init_gain, n_train, epochs=8,
                lr=3e-4, device="cpu"):
    x, y, reg, audits = load("train", limit=n_train)
    dx, dy, dreg, daudits = load("oracle_dev", limit=20_000)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    dlegal = np.asarray([bool(a.get("release_legal", True)) for a in daudits])
    mask = legal if legal_only else np.ones(len(y), dtype=bool)
    xm, ym, regm = x[mask], y[mask], reg[mask]
    # sample weights: legal=1, nonlegal=nonlegal_weight (or skip nonlegal)
    if legal_only:
        sample_w = np.ones(len(ym))
    else:
        sample_w = np.where(legal, 1.0, nonlegal_weight)
    positive = max(1, int((ym * sample_w).sum()))
    negative = max(1, int(((1 - ym) * sample_w).sum()))
    cls_weight = torch.as_tensor([1.0, negative / positive], dtype=torch.float32)

    from clinical_target_conditioned_policy import TargetConditionedClampPolicy
    from tests.test_stage1_v102 import rectangle, v102_config
    bc = FrozenBCMacroTargetPolicy(BC_MODEL, device=device)
    clinical, reward = v102_config()
    model = m._build_frozen_base_clamp_model(
        seed=2026090201, device=device, target_policy=bc,
        scenario=rectangle(rows=6, cols=6), clinical_config=clinical,
        reward_config=reward, ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    if init_gain != 0.01:
        with torch.no_grad():
            def reinit(mod):
                if isinstance(mod, torch.nn.Linear):
                    torch.nn.init.orthogonal_(mod.weight, gain=init_gain)
                    if mod.bias is not None:
                        mod.bias.zero_()
            model.policy.action_net.apply(reinit)
            model.policy.action_net.initialize_release(-4.0)
    opt = torch.optim.Adam(
        list(model.policy.action_net.parameters())
        + list(model.policy.features_extractor.plan_spatial.parameters())
        + list(model.policy.regression_head.parameters()),
        lr=lr,
    )
    sw = torch.as_tensor(sample_w, dtype=torch.float32)
    rng = np.random.default_rng(2026090201)
    batch = 256
    print(f"\n=== {name}: legal_only={legal_only} nonlegal_w={nonlegal_weight} "
          f"init_gain={init_gain} n={len(ym)} ===")
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(ym))
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            obs = torch.as_tensor(xm[b], device=device)
            target = torch.as_tensor(ym[b], dtype=torch.long, device=device)
            dt = torch.as_tensor(regm[b], dtype=torch.float32, device=device)
            w = sw[b].to(device)
            features = model.policy.extract_features(obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, obs)
            logits = model.policy.action_net.scorer(fused)
            reg_out = model.policy.regression_head(fused)
            cls = torch.nn.functional.cross_entropy(logits, target, weight=cls_weight.to(device),
                                                    reduction="none")
            cls = (cls * w).mean()
            rl = torch.nn.functional.mse_loss(reg_out, dt)
            loss = cls + 0.1 * rl
            opt.zero_grad(); loss.backward(); opt.step()
        dp = m._release_probabilities(model, dx[dlegal], device)
        dv = roc_auc_score(dy[dlegal], dp)
        trp = m._release_probabilities(model, xm[legal[mask] if legal_only else mask], device)
        tr = roc_auc_score(ym, trp) if (ym.sum() > 0 and (ym == 0).sum() > 0) else float("nan")
        print(f"  epoch {epoch}: train_auroc={tr:.4f} dev_auroc={dv:.4f} "
              f"frac_p>=.5={(dp >= 0.5).mean():.3f}")


if __name__ == "__main__":
    run_variant("V1 all", legal_only=False, nonlegal_weight=1.0, init_gain=0.01, n_train=16_000)
    run_variant("V2 legal-only", legal_only=True, nonlegal_weight=1.0, init_gain=0.01, n_train=16_000)
    run_variant("V3 legal-upweight", legal_only=False, nonlegal_weight=0.1, init_gain=0.01, n_train=16_000)
    run_variant("V4 legal+gain1", legal_only=True, nonlegal_weight=1.0, init_gain=1.0, n_train=16_000)
