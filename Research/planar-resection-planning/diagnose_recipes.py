"""Training-recipe search: can a regularized MLP / linear readout reach the
linear-probe bound (~0.70 dev AUROC) on the existing frozen fused features?

Recipes (all on train LEGAL samples, dev evaluated on oracle-dev legal):
  R1 gain=0.01 + wd=1e-3 + lr=1e-4            (regularize the underfitter)
  R2 gain=0.3  + wd=1e-3 + lr=1e-4            (mid init, regularized)
  R3 gain=1.0  + wd=1e-2 + lr=1e-4            (regularize the overfitter)
  R4 linear readout (173->2) + wd=1e-3 + lr=1e-3  (mirror the sklearn probe)

Best-epoch selection by dev AUROC (same rule as train_stage1), then report
balanced acc / release recall / unsafe-FPR at the max-balacc threshold.
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score  # noqa: E402

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


def run_recipe(tag, *, init_gain, wd, lr, linear_readout, n_train=20_000, epochs=12,
               dev_limit=6000, reg_weight=1.0, device="cpu"):
    x, y, reg, audits = load("train", limit=n_train)
    dx, dy, dreg, daudits = load("oracle_dev", limit=dev_limit)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    dlegal = np.asarray([bool(a.get("release_legal", True)) for a in daudits])
    xm, ym, regm = x[legal], y[legal], reg[legal]
    positive = max(1, int(ym.sum()))
    negative = max(1, int(len(ym) - ym.sum()))
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
    if linear_readout:
        # Replace the scorer MLP with a single linear layer (mirror probe).
        scorer = model.policy.action_net.scorer
        in_dim = scorer[0].in_features
        model.policy.action_net.scorer = torch.nn.Linear(in_dim, 2)
        model.policy.action_net.initialize_release(-4.0)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(model.policy.action_net.scorer.weight, gain=1.0)
            model.policy.action_net.scorer.bias.zero_()
            model.policy.action_net.scorer.bias[1] = -4.0
    elif init_gain != 0.01:
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
        lr=lr, weight_decay=wd,
    )
    rng = np.random.default_rng(2026090201)
    batch = 256
    print(f"\n=== {tag}: gain={init_gain} wd={wd} lr={lr} linear={linear_readout} n={len(ym)} ===", flush=True)
    best_epoch, best_auroc = 0, -1.0
    probs_by_epoch = {}
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(ym))
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            obs = torch.as_tensor(xm[b], device=device)
            target = torch.as_tensor(ym[b], dtype=torch.long, device=device)
            dt = torch.as_tensor(regm[b], dtype=torch.float32, device=device)
            features = model.policy.extract_features(obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, obs)
            logits = model.policy.action_net.scorer(fused)
            reg_out = model.policy.regression_head(fused)
            cls = torch.nn.functional.cross_entropy(logits, target, weight=cls_weight.to(device))
            rl = torch.nn.functional.mse_loss(reg_out, dt)
            loss = cls + reg_weight * rl
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            dp = m._release_probabilities(model, dx[dlegal], device)
            dv = roc_auc_score(dy[dlegal], dp)
            probs_by_epoch[epoch] = dp
            print(f"  epoch {epoch:02d}: dev_auroc={dv:.4f} frac_p>=.5={(dp>=0.5).mean():.3f}", flush=True)
            if dv > best_auroc:
                best_auroc, best_epoch = dv, epoch
    dp_best = probs_by_epoch[best_epoch]
    # threshold selection by max balanced accuracy on dev (same rule as train_stage1)
    best_thr, best_bal = 0.5, -1.0
    for t in np.round(np.arange(0.10, 0.95, 0.05), 2):
        pred = (dp_best >= t).astype(int)
        bal = balanced_accuracy_score(dy[dlegal], pred)
        if bal > best_bal:
            best_bal, best_thr = bal, float(t)
    pred = (dp_best >= best_thr).astype(int)
    unsafe = dy[dlegal] == 0
    unsafe_fpr = float((pred[unsafe] == 1).mean()) if unsafe.sum() else 0.0
    rec = recall_score(dy[dlegal], pred, zero_division=0)
    print(f"  BEST epoch {best_epoch}: dev_auroc={best_auroc:.4f} "
          f"best_thr={best_thr} bal_acc={best_bal:.4f} release_recall={rec:.4f} "
          f"unsafe_fpr={unsafe_fpr:.4f}", flush=True)


if __name__ == "__main__":
    run_recipe("R1 underfit+wd", init_gain=0.01, wd=1e-3, lr=1e-4, linear_readout=False)
    run_recipe("R2 mid+wd", init_gain=0.3, wd=1e-3, lr=1e-4, linear_readout=False)
    run_recipe("R3 overfit+wd", init_gain=1.0, wd=1e-2, lr=1e-4, linear_readout=False)
    run_recipe("R4 linear", init_gain=0.01, wd=1e-3, lr=1e-3, linear_readout=True)
