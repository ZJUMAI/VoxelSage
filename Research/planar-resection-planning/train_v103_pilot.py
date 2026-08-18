"""v10.3 pilot: predict Delta-B / Delta-I from look-ahead features and gate
release with a conservative safety head.

Pipeline (scene-isolated internal split, frozen rules):
  1. load pilot features on internal train / calibration / dev;
  2. standardize on train; train a small MLP (shared trunk, two heads) to
     regress normalized Delta-B and Delta-I (MSE);
  3. estimate residual stds sigma_b / sigma_i on the CALIBRATION set;
  4. safety score = Phi(-pred_db/sigma_b) * Phi(-pred_di/sigma_i);
  5. tune the release threshold on CALIBRATION to maximize recall subject to
     unsafe-FPR <= 5%;
  6. evaluate the four admission bars on INTERNAL DEV (untouched during
     tuning): AUROC >= 0.75, balanced acc >= 0.70, release recall >= 0.50,
     unsafe-FPR <= 0.05.

Usage:
    python train_v103_pilot.py --outdir results/clinical_window_v10_2/pilot_v103
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from v103_lookahead_features import FEATURE_NAMES  # noqa: E402

DATA_DIR = Path("results/clinical_window_v10_2/pilot_v103")


class DualHeadMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_b = nn.Linear(hidden, 1)
        self.head_i = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_b(h).squeeze(-1), self.head_i(h).squeeze(-1)


def load_all():
    manifest = json.loads((DATA_DIR / "internal_split.json").read_text(encoding="utf-8"))
    feats, reg, db, di, labels = [], [], [], [], []
    for npz in sorted(DATA_DIR.glob("pilot_*.npz")):
        d = np.load(npz)
        feats.append(d["features"]); reg.append(d["regression"])
        db.append(d["delta_blood"]); di.append(d["delta_ischemia"])
        labels.append(d["labels"])
    all_audits = []
    for af in sorted(DATA_DIR.glob("audit_*.json")):
        all_audits.extend(json.loads(af.read_text(encoding="utf-8"))["examples"])
    return (
        np.concatenate(feats).astype(np.float32),
        np.concatenate(reg).astype(np.float32),
        np.concatenate(db).astype(np.float32),
        np.concatenate(di).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
        all_audits,
        manifest,
    )


def filter_split(all_f, all_reg, all_db, all_di, all_lbl, all_aud, ids):
    m = np.asarray([a["scenario_id"] in ids for a in all_aud], dtype=bool)
    return (all_f[m], all_reg[m], all_db[m], all_di[m], all_lbl[m])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    all_f, all_reg, all_db, all_di, all_lbl, all_aud, manifest = load_all()
    tr_ids = set(manifest["internal_train"])
    ca_ids = set(manifest["internal_calibration"])
    de_ids = set(manifest["internal_dev"])
    tr_f, tr_reg, tr_db, tr_di, tr_lbl = filter_split(all_f, all_reg, all_db, all_di, all_lbl, all_aud, tr_ids)
    ca_f, ca_reg, ca_db, ca_di, ca_lbl = filter_split(all_f, all_reg, all_db, all_di, all_lbl, all_aud, ca_ids)
    de_f, de_reg, de_db, de_di, de_lbl = filter_split(all_f, all_reg, all_db, all_di, all_lbl, all_aud, de_ids)

    print(f"internal_train n={len(tr_lbl)} pos={int(tr_lbl.sum())} "
          f"cal n={len(ca_lbl)} dev n={len(de_lbl)} pos={int(de_lbl.sum())}", flush=True)

    scaler = StandardScaler()
    scaler.fit(tr_f)
    tr_f = scaler.transform(tr_f).astype(np.float32)
    ca_f = scaler.transform(ca_f).astype(np.float32)
    de_f = scaler.transform(de_f).astype(np.float32)

    torch.manual_seed(args.seed)
    model = DualHeadMLP(tr_f.shape[1], hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_x = torch.as_tensor(tr_f); tr_b = torch.as_tensor(tr_reg[:, 0]); tr_i = torch.as_tensor(tr_reg[:, 1])
    ca_x = torch.as_tensor(ca_f); de_x = torch.as_tensor(de_f)
    rng = np.random.default_rng(args.seed)
    n = len(tr_lbl)

    # weight regression targets equally: delta-B and delta-I scales already normalized
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = rng.permutation(n)
        for start in range(0, n, args.batch):
            b = idx[start:start + args.batch]
            pred_b, pred_i = model(tr_x[b])
            loss = nn.functional.mse_loss(pred_b, tr_b[b]) + nn.functional.mse_loss(pred_i, tr_i[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_ca_b, pred_ca_i = model(ca_x)
        pred_de_b, pred_de_i = model(de_x)
    pred_ca_b = pred_ca_b.numpy(); pred_ca_i = pred_ca_i.numpy()
    pred_de_b = pred_de_b.numpy(); pred_de_i = pred_de_i.numpy()

    # residual stds on calibration
    sigma_b = float(np.sqrt(np.mean((pred_ca_b - ca_reg[:, 0]) ** 2)))
    sigma_i = float(np.sqrt(np.mean((pred_ca_i - ca_reg[:, 1]) ** 2)))
    print(f"sigma_b={sigma_b:.4f} sigma_i={sigma_i:.4f}", flush=True)

    def safety_score(db_pred, di_pred):
        from scipy.stats import norm
        pb = norm.cdf(-db_pred / max(sigma_b, 1e-6))
        pi = norm.cdf(-di_pred / max(sigma_i, 1e-6))
        return pb * pi

    ca_score = safety_score(pred_ca_b, pred_ca_i)
    de_score = safety_score(pred_de_b, pred_de_i)

    def g1_metrics(y, db_true, di_true, pred):
        unsafe = (db_true > 0) | (di_true >= 0)
        n_unsafe = int(unsafe.sum())
        return {
            "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "release_recall": float(recall_score(y, pred, zero_division=0)),
            "unsafe_fpr": float((pred.astype(bool) & unsafe).sum() / n_unsafe) if n_unsafe else 0.0,
            "n_unsafe": n_unsafe,
        }

    def release_rule(db_pred, di_pred, k):
        # Conservative safety head: release only when the predicted blood loss
        # upper confidence bound is <= 0 and ischemia is also beneficial.
        return ((db_pred + k * sigma_b) <= 0) & ((di_pred + k * sigma_i) < 0)

    # Tune the shared confidence margin k on CALIBRATION: max release recall
    # subject to unsafe-FPR <= 5%.
    best_k, best_recall = None, -1.0
    for k in np.round(np.arange(0.0, 5.01, 0.05), 2):
        pred = release_rule(pred_ca_b, pred_ca_i, float(k)).astype(int)
        m = g1_metrics(ca_lbl, ca_db, ca_di, pred)
        if m["unsafe_fpr"] <= 0.05 and m["release_recall"] > best_recall:
            best_recall, best_k = m["release_recall"], float(k)
    if best_k is None:
        print("NO k on calibration satisfies unsafe-FPR<=5%", flush=True)
        best_k = 0.0

    dev_pred = release_rule(pred_de_b, pred_de_i, best_k).astype(int)
    dev_met = g1_metrics(de_lbl, de_db, de_di, dev_pred)
    dev_met["auroc"] = float(roc_auc_score(de_lbl, de_score))
    cal_pred = release_rule(pred_ca_b, pred_ca_i, best_k).astype(int)
    cal_met = g1_metrics(ca_lbl, ca_db, ca_di, cal_pred)
    bars = {
        "unsafe_fpr_le_005": dev_met["unsafe_fpr"] <= 0.05,
        "release_recall_ge_050": dev_met["release_recall"] >= 0.50,
        "auroc_ge_075": dev_met["auroc"] >= 0.75,
        "balanced_acc_ge_070": dev_met["balanced_acc"] >= 0.70,
    }
    print(json.dumps({
        "mode": "train_v103_pilot",
        "sigma_b": sigma_b, "sigma_i": sigma_i,
        "tuned_margin_k": best_k,
        "calibration": {k: round(v, 4) for k, v in cal_met.items() if isinstance(v, float)},
        "internal_dev": {k: round(v, 4) for k, v in dev_met.items() if isinstance(v, float)},
        "bars": bars,
        "decision": "PASS" if all(bars.values()) else "FAIL",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
