"""Diagnose v10.3 Delta-B / Delta-I regression quality.

Checks: target distribution, R2 of a linear readout and the MLP on train/cal,
and whether MSE is dominated by outliers.
"""
import json, sys
from pathlib import Path
import numpy as np
SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
import torch, torch.nn as nn
from sklearn.linear_model import LinearRegression

DATA = Path("results/clinical_window_v10_2/pilot_v103")
manifest = json.loads((DATA / "internal_split.json").read_text())
feats, reg, db, di, labels, audits = [], [], [], [], [], []
for f in sorted(DATA.glob("pilot_*.npz")):
    d = np.load(f)
    feats.append(d["features"]); reg.append(d["regression"])
    db.append(d["delta_blood"]); di.append(d["delta_ischemia"]); labels.append(d["labels"])
for af in sorted(DATA.glob("audit_*.json")):
    audits.extend(json.load(open(af))["examples"])
F = np.concatenate(feats).astype(np.float32); R = np.concatenate(reg).astype(np.float32)
DB = np.concatenate(db).astype(np.float32); DI = np.concatenate(di).astype(np.float32)
L = np.concatenate(labels).astype(np.int64)

def mask(ids):
    return np.asarray([a["scenario_id"] in ids for a in audits], dtype=bool)
tr_m = mask(set(manifest["internal_train"]))
ca_m = mask(set(manifest["internal_calibration"]))
de_m = mask(set(manifest["internal_dev"]))

# target distribution
print("delta_blood (mL): min=%.1f max=%.1f mean=%.1f std=%.1f p1=%.1f p99=%.1f" %
      (DB.min(), DB.max(), DB.mean(), DB.std(), np.percentile(DB,1), np.percentile(DB,99)))
print("delta_ischemia (min): min=%.2f max=%.2f mean=%.2f std=%.2f" %
      (DI.min(), DI.max(), DI.mean(), DI.std()))
print("reg targets: db_norm mean=%.4f std=%.4f | di_norm mean=%.4f std=%.4f" %
      (R[:,0].mean(), R[:,0].std(), R[:,1].mean(), R[:,1].std()))

# linear R2
for name, m_, reg_, lab in [("train", tr_m, R, L), ("cal", ca_m, R, L), ("dev", de_m, R, L)]:
    X = F[m_]; yb = reg_[m_, 0]; yi = reg_[m_, 1]
    lr_b = LinearRegression().fit(X, yb); lr_i = LinearRegression().fit(X, yi)
    r2_b = lr_b.score(X, yb); r2_i = lr_i.score(X, yi)
    print(f"linear R2 {name}: db={r2_b:.4f} di={r2_i:.4f}")

# simple MLP R2 (train, then eval on all)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(F[tr_m])
Xtr = torch.as_tensor(sc.transform(F[tr_m]).astype(np.float32))
ybtr = torch.as_tensor(R[tr_m, 0]); yitr = torch.as_tensor(R[tr_m, 1])
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(F.shape[1],64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(),
                    nn.Linear(64,2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
idx = np.random.default_rng(0).permutation(len(Xtr))
for epoch in range(40):
    for s in range(0, len(idx), 512):
        b = idx[s:s+512]
        out = net(Xtr[b])
        loss = nn.functional.mse_loss(out[:,0], ybtr[b]) + nn.functional.mse_loss(out[:,1], yitr[b])
        opt.zero_grad(); loss.backward(); opt.step()
def r2_mlp(m_):
    X = sc.transform(F[m_]).astype(np.float32)
    with torch.no_grad():
        out = net(torch.as_tensor(X))
    p_b, p_i = out[:,0].numpy(), out[:,1].numpy()
    rb = 1 - ((R[m_,0]-p_b)**2).mean()/((R[m_,0]-R[m_,0].mean())**2).mean()
    ri = 1 - ((R[m_,1]-p_i)**2).mean()/((R[m_,1]-R[m_,1].mean())**2).mean()
    return rb, ri
for name, m_ in [("train", tr_m), ("cal", ca_m), ("dev", de_m)]:
    rb, ri = r2_mlp(m_)
    print(f"MLP R2 {name}: db={rb:.4f} di={ri:.4f}")

# does rw_blood_loss alone predict db_norm? (the key look-ahead feature)
rw_all = np.asarray([a["rw_blood_loss"] for a in audits], dtype=float)
for name, m_ in [("train", tr_m), ("cal", ca_m), ("dev", de_m)]:
    r = np.corrcoef(rw_all[m_], R[m_,0])[0,1]
    print(f"corr(rw_blood_loss, db_norm) {name}: r={r:.4f}")
