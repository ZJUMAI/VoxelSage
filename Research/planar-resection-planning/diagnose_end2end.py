"""End-to-end CNN on the raw 36-channel obs: can a trainable spatial model
beat the ~0.67 frozen-feature ceiling?

Variants:
  E1  full 36-channel obs, small CNN, all trainable
  E2  base 26 channels only (frozen design's base branch made trainable)

If E1/E2 exceed ~0.70 dev AUROC, the frozen-feature design is the bottleneck
and unfreezing is a viable design change.  If they also cap near 0.67, the
label is fundamentally hard given the obs.
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
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DATA_DIR = Path("results/clinical_window_v10_2/stage1_data")


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


class SmallCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.head(self.net(x))


def g1_metrics(y, p, audit, threshold=0.5):
    yt, yp = y, p
    pred = (yp >= threshold).astype(int)
    db = np.asarray([float(a.get("delta_blood", 0.0)) for a in audit])
    di = np.asarray([float(a.get("delta_ischemia", 0.0)) for a in audit])
    unsafe = (db > 0) | (di >= 0)
    from sklearn.metrics import balanced_accuracy_score, recall_score
    return {
        "auroc": float(roc_auc_score(yt, yp)),
        "balanced_acc": float(balanced_accuracy_score(yt, pred)),
        "release_recall": float(recall_score(yt, pred, zero_division=0)),
        "unsafe_fpr": float((pred[unsafe] == 1).mean()) if unsafe.sum() else 0.0,
    }


def run(tag, in_ch, n_train, epochs, device, batch=256, lr=1e-3, wd=1e-3):
    x, y, audits = load("train", limit=n_train)
    dx, dy, daudits = load("oracle_dev", limit=14_000)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audits])
    dlegal = np.asarray([bool(a.get("release_legal", True)) for a in daudits])
    xl, yl = x[legal, :in_ch], y[legal]
    dxl, dyl = dx[dlegal, :in_ch], dy[dlegal]
    dlegal_list = [a for a, ok in zip(daudits, dlegal) if ok]
    positive = max(1, int(yl.sum())); negative = max(1, int(len(yl) - yl.sum()))
    cw = torch.as_tensor([1.0, negative / positive], dtype=torch.float32, device=device)

    torch.manual_seed(2026090201)
    model = SmallCNN(in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(2026090201)
    print(f"\n=== {tag}: in_ch={in_ch} n={len(yl)} ===", flush=True)
    best = -1.0
    best_pva = None
    for epoch in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(len(yl))
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            obs = torch.as_tensor(xl[b], device=device)
            target = torch.as_tensor(yl[b], dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(model(obs), target, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ps = []
            for start in range(0, len(dxl), 512):
                p = torch.softmax(model(torch.as_tensor(dxl[start:start + 512], device=device)), 1)
                ps.append(p[:, 1].cpu().numpy())
        pva = np.concatenate(ps)
        auroc = roc_auc_score(dyl, pva)
        if auroc > best:
            best, best_pva = auroc, pva
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:02d}: dev_auroc={auroc:.4f} frac_p>=.5={(pva>=0.5).mean():.3f}", flush=True)
    # G1-style metrics at the best-AUROC epoch (threshold by max dev bal acc)
    from sklearn.metrics import balanced_accuracy_score
    yt, yp = dyl, best_pva
    best_t, best_bal = 0.5, -1.0
    for t in np.round(np.arange(0.10, 0.95, 0.05), 2):
        bal = balanced_accuracy_score(yt, (yp >= t).astype(int))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    met = g1_metrics(dyl, best_pva, dlegal_list, threshold=best_t)
    print(f"  BEST dev_auroc={best:.4f} thr={best_t} bal_acc={met['balanced_acc']:.4f} "
          f"recall={met['release_recall']:.4f} unsafe_fpr={met['unsafe_fpr']:.4f}", flush=True)


if __name__ == "__main__":
    device = "cuda"
    run("E1 full36", in_ch=36, n_train=20_000, epochs=12, device=device)
    run("E2 base26", in_ch=26, n_train=20_000, epochs=12, device=device)
