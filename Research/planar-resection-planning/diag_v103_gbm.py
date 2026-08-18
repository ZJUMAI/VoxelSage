"""Feature-sufficiency check: HistGBM on the 22 look-ahead features.

If a strong non-linear model also caps near AUROC 0.72 / recall ~9%, the
features are the bottleneck; if it jumps well above, the MLP was the issue.
Also prints per-feature correlation with db_norm / di_norm.
"""
import json, sys
from pathlib import Path
import numpy as np
SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from v103_lookahead_features import FEATURE_NAMES
from scipy.stats import norm

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
tr_m = mask(set(manifest["internal_train"])); ca_m = mask(set(manifest["internal_calibration"]))
de_m = mask(set(manifest["internal_dev"]))

print("== per-feature correlation with db_norm / di_norm (on dev legal) ==")
for i, name in enumerate(FEATURE_NAMES):
    rd = np.corrcoef(F[de_m, i], R[de_m, 0])[0,1]
    ri = np.corrcoef(F[de_m, i], R[de_m, 1])[0,1]
    print(f"  {name:28s} r_db={rd:+.3f}  r_di={ri:+.3f}")

sc = StandardScaler().fit(F[tr_m])
Xtr = sc.transform(F[tr_m]); Xca = sc.transform(F[ca_m]); Xde = sc.transform(F[de_m])
print("\n== HistGBM regression ==")
pred = {}
for tag, col in [("db", 0), ("di", 1)]:
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                          random_state=0, validation_fraction=0.15,
                                          early_stopping=True)
    model.fit(Xtr, R[tr_m, col])
    for name, X, m_ in [("train", Xtr, tr_m), ("cal", Xca, ca_m), ("dev", Xde, de_m)]:
        p = model.predict(X)
        r2 = 1 - ((R[m_,col]-p)**2).mean() / ((R[m_,col]-R[m_,col].mean())**2).mean()
        pred[(tag, name)] = p
        if name == "dev":
            print(f"  {tag} dev R2={r2:.4f}")
    # correlation of prediction with true on dev
    print(f"  {tag} dev corr(pred,true)={np.corrcoef(pred[(tag,'dev')], R[de_m,col])[0,1]:.4f}")

# safety head with GBM predictions
pb = pred[("db","dev")]; pi = pred[("di","dev")]
cb = pred[("db","cal")]; ci = pred[("di","cal")]
sb_cal = np.std(cb - R[ca_m,0]); si_cal = np.std(ci - R[ca_m,1])
print(f"\nsigma_b={sb_cal:.4f} sigma_i={si_cal:.4f}")
def score(db, di, sb, si):
    return norm.cdf(-db/sb) * norm.cdf(-di/si)
s_ca = score(cb, ci, sb_cal, si_cal); s_de = score(pb, pi, sb_cal, si_cal)
auroc = roc_auc_score(L[de_m], s_de)
print(f"GBM safety-score dev AUROC={auroc:.4f}")
def rule(db, di, k, sb, si):
    return ((db + k*sb) <= 0) & ((di + k*si) < 0)
best_k, best_recall = None, -1.0
for k in np.round(np.arange(0.0, 5.01, 0.05), 2):
    pr = rule(cb, ci, k, sb_cal, si_cal).astype(int)
    unsafe = (DB[ca_m] > 0) | (DI[ca_m] >= 0)
    fpr = (pr.astype(bool) & unsafe).sum()/max(1,unsafe.sum())
    rec = recall_score(L[ca_m], pr, zero_division=0)
    if fpr <= 0.05 and rec > best_recall:
        best_recall, best_k = rec, k
print(f"tuned k={best_k} (cal recall={best_recall:.4f})")
pr = rule(pb, pi, best_k, sb_cal, si_cal).astype(int)
unsafe = (DB[de_m] > 0) | (DI[de_m] >= 0)
fpr = (pr.astype(bool) & unsafe).sum()/max(1,unsafe.sum())
rec = recall_score(L[de_m], pr, zero_division=0)
bal = balanced_accuracy_score(L[de_m], pr)
print(f"dev: recall={rec:.4f} unsafe_fpr={fpr:.4f} bal_acc={bal:.4f}")
