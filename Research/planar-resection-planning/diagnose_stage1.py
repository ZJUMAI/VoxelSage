"""Stage-1 non-convergence diagnostics (AUROC stuck ~0.52).

Checks:
  1. Is the v2-safe label learnable from the observation?  Correlate the
     label with every scalar (fill-constant) channel on the release-legal
     subset; run a sklearn logistic probe on scalars + target one-hot to get
     an achievable AUROC bound.
  2. Does a freshly-initialized model move off its init at all?  Distribution
     of release probabilities on dev legal samples.
  3. Per-batch training: cls_loss / reg_loss / plan_spatial + scorer grad
     norms, so we can tell "optimization dead" from "label uninformative".
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score  # noqa: E402

from clinical_target_conditioned_environment import CLAMP_RELEASE  # noqa: E402
from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy  # noqa: E402

import train_target_conditioned_clamp_oracle as m  # noqa: E402

DATA_DIR = Path("results/clinical_window_v10_2/stage1_data")
BC_MODEL = "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"
SCALAR_CH = list(range(28, 36)) + [17, 18, 19, 20, 24]


def load_dev_subset(limit: int = 20_000):
    obs_parts, y_parts, reg_parts, audits = [], [], [], []
    seen = 0
    for npz in sorted(DATA_DIR.glob("stage1_oracle_dev_*.npz")):
        d = np.load(npz)
        obs_parts.append(d["obs"][: limit - seen])
        y_parts.append(d["labels"][: limit - seen])
        reg_parts.append(d["regression"][: limit - seen])
        seen += len(obs_parts[-1])
        if seen >= limit:
            break
    for a in sorted(DATA_DIR.glob("audit_oracle_dev_*.json")):
        audits.extend(json.loads(a.read_text(encoding="utf-8"))["audit"])
        if len(audits) >= limit:
            break
    return (
        np.concatenate(obs_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int64),
        np.concatenate(reg_parts).astype(np.float32),
        audits[: seen],
    )


def check_label_predictability(x, y, audits):
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    xl, yl = x[legal], y[legal]
    print(f"\n== label predictability on legal subset ==")
    print(f"n_legal={len(yl)}  n_positive={int(yl.sum())}  pos_frac={yl.mean():.4f}")
    # 1) per-scalar mean difference pos vs neg
    print("\nscalar channel means (pos vs neg)  [ch: pos | neg | std_diff/sqrt(n)]:")
    for ch in SCALAR_CH:
        vals = xl[:, ch, 0, 0]
        pos_m, neg_m = vals[yl == 1].mean(), vals[yl == 0].mean()
        diff = pos_m - neg_m
        pooled = np.sqrt(vals[yl == 1].var() + vals[yl == 0].var())
        print(f"  ch{ch:02d}: {pos_m:+.3f} | {neg_m:+.3f} | d={diff:+.3f}  pooled_sd={pooled:.3f}")
    # 2) logistic probe on scalars + target one-hot + route density
    feat = np.concatenate([
        xl[:, SCALAR_CH, 0, 0],
        xl[:, 26].reshape(len(yl), -1)[:, :: 40 * 5],  # sparse target probe
    ], axis=1)
    idx = np.arange(len(yl))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    split = int(0.7 * len(idx))
    tr, va = idx[:split], idx[split:]
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(feat[tr], yl[tr])
    p = clf.predict_proba(feat[va])[:, 1]
    auroc = roc_auc_score(yl[va], p)
    bal = balanced_accuracy_score(yl[va], (p >= 0.5).astype(int))
    print(f"\nlogistic probe (scalars + sparse target): dev AUROC={auroc:.4f} bal_acc={bal:.4f}")
    # 3) "release-saves-ischemia" style oracle: label vs delta_ischemia
    di = np.asarray([float(a.get("delta_ischemia", 0.0)) for a in audits])[legal]
    db = np.asarray([float(a.get("delta_blood", 0.0)) for a in audits])[legal]
    print(f"delta_ischemia: pos mean={di[yl==1].mean():+.4f} neg mean={di[yl==0].mean():+.4f}")
    print(f"delta_blood: pos mean={db[yl==1].mean():+.4f} neg mean={db[yl==0].mean():+.4f} "
          f"(legal-only, expect ~0 both sides if safe_release_mask kills exposure)")
    # 4) how often is delta_ischemia strictly negative?
    print(f"legal samples with delta_ischemia < -1e-6: {(di < -1e-6).mean():.4f} "
          f"(label should == that if db<=0 always)")


def check_init_prediction(x, y, audits, device="cpu"):
    print("\n== fresh-init model prediction distribution (dev legal) ==")
    from clinical_target_conditioned_policy import TargetConditionedClampPolicy
    from tests.test_stage1_v102 import rectangle, v102_config
    bc = FrozenBCMacroTargetPolicy(BC_MODEL, device=device)
    clinical, reward = v102_config()
    scenario = rectangle(rows=6, cols=6)
    model = m._build_frozen_base_clamp_model(
        seed=7, device=device, target_policy=bc, scenario=scenario,
        clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    prob = m._release_probabilities(model, x[legal], device)
    yl = y[legal]
    print(f"release prob: min={prob.min():.4f} p25={np.percentile(prob,25):.4f} "
          f"mean={prob.mean():.4f} p75={np.percentile(prob,75):.4f} max={prob.max():.4f}")
    print(f"frac prob>=0.5: {(prob>=0.5).mean():.4f}   frac prob>=0.1: {(prob>=0.1).mean():.4f}")
    print(f"AUROC at init = {roc_auc_score(yl, prob):.4f}")


def quick_train(x, y, reg, audits, device="cpu", epochs=2, batch=256, lr=1e-4):
    print("\n== quick train (2 epochs, full loss breakdown, grad norms) ==")
    from clinical_target_conditioned_policy import TargetConditionedClampPolicy
    from tests.test_stage1_v102 import rectangle, v102_config
    bc = FrozenBCMacroTargetPolicy(BC_MODEL, device=device)
    clinical, reward = v102_config()
    scenario = rectangle(rows=6, cols=6)
    model = m._build_frozen_base_clamp_model(
        seed=2026090201, device=device, target_policy=bc, scenario=scenario,
        clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=20.0,
    )
    opt = torch.optim.Adam(
        list(model.policy.action_net.parameters())
        + list(model.policy.features_extractor.plan_spatial.parameters())
        + list(model.policy.regression_head.parameters()),
        lr=lr,
    )
    positive = max(1, int(y.sum()))
    negative = max(1, int(len(y) - y.sum()))
    cw = torch.as_tensor([1.0, negative / positive], dtype=torch.float32, device=device)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    rng = np.random.default_rng(2026090201)
    for epoch in range(epochs):
        idx = rng.permutation(len(y))
        cls_hist, reg_hist = [], []
        gplan_hist, gscorer_hist, greg_hist = [], [], []
        for start in range(0, len(idx), batch):
            b = idx[start:start+batch]
            obs = torch.as_tensor(x[b], device=device)
            target = torch.as_tensor(y[b], dtype=torch.long, device=device)
            dt = torch.as_tensor(reg[b], dtype=torch.float32, device=device)
            features = model.policy.extract_features(obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, obs)
            logits = model.policy.action_net.scorer(fused)
            reg_out = model.policy.regression_head(fused)
            cls_loss = torch.nn.functional.cross_entropy(logits, target, weight=cw)
            reg_loss = torch.nn.functional.mse_loss(reg_out, dt)
            loss = cls_loss + 0.1 * reg_loss
            opt.zero_grad()
            loss.backward()
            gp = torch.cat([p.grad.flatten() for p in model.policy.features_extractor.plan_spatial.parameters()])
            gs = torch.cat([p.grad.flatten() for p in model.policy.action_net.parameters()])
            gr = torch.cat([p.grad.flatten() for p in model.policy.regression_head.parameters()])
            gplan_hist.append(float(gp.norm()))
            gscorer_hist.append(float(gs.norm()))
            greg_hist.append(float(gr.norm()))
            opt.step()
            cls_hist.append(float(cls_loss.detach().cpu()))
            reg_hist.append(float(reg_loss.detach().cpu()))
        print(f"epoch {epoch}: cls={np.mean(cls_hist):.4f} reg={np.mean(reg_hist):.4f} "
              f"|grad| plan={np.mean(gplan_hist):.3f} scorer={np.mean(gscorer_hist):.3f} reghead={np.mean(greg_hist):.3f}")
    prob = m._release_probabilities(model, x[legal], device)
    print(f"post-train dev AUROC={roc_auc_score(y[legal], prob):.4f} "
          f"frac prob>=0.5={(prob>=0.5).mean():.4f}")


if __name__ == "__main__":
    x, y, reg, audits = load_dev_subset()
    print(f"loaded dev subset: obs={x.shape} dtype={x.dtype}")
    check_label_predictability(x, y, audits)
    check_init_prediction(x, y, audits)
    quick_train(x, y, reg, audits)
