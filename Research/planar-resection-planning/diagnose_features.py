"""Is the model's fused feature vector actually DISCRIMINATIVE at legal points?

Hypothesis: at release-legal points (safe_release_mask -> no exposed vessels,
phase_elapsed >= early_end), the frozen BC base features and the plan/scalar
channels are nearly constant across samples, so the scorer input has almost no
variance and AUROC is pinned near 0.5.

Measurements (dev legal subset):
  1. variance + range of the 173-dim fused feature across samples;
  2. mean pairwise cosine similarity of fused features;
  3. same for the base global-pooled features alone;
  4. correlation between fused-feature PCA-1 and the label.
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

from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy  # noqa: E402

import train_target_conditioned_clamp_oracle as m  # noqa: E402

DATA_DIR = Path("results/clinical_window_v10_2/stage1_data")
BC_MODEL = "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"
SCALAR_CH = list(range(28, 36)) + [17, 18, 19, 20, 24]


def load_dev(limit=20_000):
    obs_parts, y_parts, audits = [], [], []
    seen = 0
    for npz in sorted(DATA_DIR.glob("stage1_oracle_dev_*.npz")):
        d = np.load(npz)
        n = min(limit - seen, len(d["labels"]))
        if n <= 0:
            break
        obs_parts.append(d["obs"][:n])
        y_parts.append(d["labels"][:n])
        seen += n
        if seen >= limit:
            break
    for a in sorted(DATA_DIR.glob("audit_oracle_dev_*.json")):
        audits.extend(json.loads(a.read_text(encoding="utf-8"))["audit"])
        if len(audits) >= seen:
            break
    return (
        np.concatenate(obs_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int64),
        audits[: seen],
    )


def fused_features(model, obs_batch, device="cpu"):
    x = torch.as_tensor(obs_batch, device=device)
    with torch.no_grad():
        features = model.policy.extract_features(x)
        latent_pi, _ = model.policy.mlp_extractor(features)
        fused = model.policy.action_net.fused_features(latent_pi, x)
    return fused.detach().cpu().numpy()


def main():
    x, y, audits = load_dev()
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    xl, yl = x[legal], y[legal]
    # subsample for speed
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(len(yl), size=min(4000, len(yl)), replace=False))
    xl, yl = xl[take], yl[take]
    print(f"dev legal subsample: n={len(yl)} pos={int(yl.sum())} ({yl.mean():.3f})")

    from clinical_target_conditioned_policy import TargetConditionedClampPolicy
    from tests.test_stage1_v102 import rectangle, v102_config
    bc = FrozenBCMacroTargetPolicy(BC_MODEL, device="cpu")
    clinical, reward = v102_config()
    model = m._build_frozen_base_clamp_model(
        seed=7, device="cpu", target_policy=bc, scenario=rectangle(rows=6, cols=6),
        clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    fused = fused_features(model, xl)
    print(f"fused shape={fused.shape}")
    per_dim_std = fused.std(axis=0)
    print(f"fused: mean_std={per_dim_std.mean():.5f} median_std={np.median(per_dim_std):.5f} "
          f"min_std={per_dim_std.min():.5f} max_std={per_dim_std.max():.5f} "
          f"frac_dead_dims={(per_dim_std < 1e-6).mean():.3f}")
    # mean pairwise cosine (sampled)
    idx = rng.choice(len(fused), size=min(300, len(fused)), replace=False)
    sub = fused[idx]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    cos = sub @ sub.T
    print(f"fused mean pairwise cosine: {np.mean(cos[np.triu_indices(len(sub), 1)]):.5f}")
    # PCA-1 correlation with label
    u, s, _ = np.linalg.svd(fused - fused.mean(0), full_matrices=False)
    pca1 = u[:, 0] * s[0]
    print(f"fused PCA-1 correlation with label: {np.corrcoef(pca1, yl)[0, 1]:.4f}")
    # label vs each scalar channel
    print("\nlabel vs scalar channel (point-biserial):")
    for ch in SCALAR_CH:
        r = np.corrcoef(xl[:, ch, 0, 0], yl)[0, 1]
        print(f"  ch{ch:02d}: r={r:+.4f}")
    # how many obs (36ch) are EXACTLY identical to another sample's obs?
    obs_flat = xl.reshape(len(xl), -1)
    uniq, counts = np.unique(obs_flat, axis=0, return_counts=True)
    print(f"\nobs: n={len(obs_flat)} unique={len(uniq)} dup_frac={1 - len(uniq)/len(obs_flat):.4f}")
    # among duplicated obs, is the label consistent?
    if len(uniq) < len(obs_flat):
        map_ = {i: c for i, c in enumerate(counts)}
        dup_rows = np.where(counts > 1)[0]
        first = {tuple(uniq[i]): None for i in dup_rows}
        inconsistent = 0
        checked = 0
        for row in range(len(xl)):
            key = tuple(obs_flat[row])
            if key in first:
                if first[key] is None:
                    first[key] = int(yl[row])
                else:
                    checked += 1
                    if first[key] != int(yl[row]):
                        inconsistent += 1
        print(f"duplicate-obs label consistency: {checked} dup pairs, {inconsistent} INCONSISTENT")


if __name__ == "__main__":
    main()
