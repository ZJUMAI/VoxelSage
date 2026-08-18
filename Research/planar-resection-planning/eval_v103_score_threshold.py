"""Re-evaluate v10.3 with a SCORE-THRESHOLD safety head (tuned on calibration).

The decision-maker's "release only when the predicted blood upper confidence
bound <= 0" is implemented as a threshold on the joint safety probability
score  P(db<=0 & di<0) = Phi(-pred_db/sigma_b) * Phi(-pred_di/sigma_i).
Threshold t is tuned on CALIBRATION (max release recall subject to
unsafe-FPR <= 5%) and the four admission bars are evaluated on INTERNAL DEV.
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
def mask(ids): return np.asarray([a["scenario_id"] in ids for a in audits], bool)
tr_m = mask(set(manifest["internal_train"])); ca_m = mask(set(manifest["internal_calibration"]))
de_m = mask(set(manifest["internal_dev"]))

sc = StandardScaler().fit(F[tr_m])
Xtr, Xca, Xde = sc.transform(F[tr_m]), sc.transform(F[ca_m]), sc.transform(F[de_m])
preds = {}
for col in (0, 1):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4,
                                      random_state=0, validation_fraction=0.15, early_stopping=True)
    m.fit(Xtr, R[tr_m, col])
    preds[col] = (m.predict(Xca), m.predict(Xde))
sb = float(np.std(preds[0][0] - R[ca_m, 0]))
si = float(np.std(preds[1][0] - R[ca_m, 1]))
print(f"sigma_b={sb:.4f} sigma_i={si:.4f}")

def score(b, i): return norm.cdf(-b / sb) * norm.cdf(-i / si)
s_ca = score(preds[0][0], preds[1][0])
s_de = score(preds[0][1], preds[1][1])

# Tune threshold t on CALIBRATION: max recall s.t. unsafe-FPR <= 5%
unsafe_ca = (DB[ca_m] > 0) | (DI[ca_m] >= 0)
safe_ca = L[ca_m] == 1
best_t, best_recall = None, -1.0
for t in np.round(np.arange(0.0, 1.0, 0.005), 3):
    rel = s_ca >= t
    fpr = rel[unsafe_ca].mean() if unsafe_ca.sum() else 0.0
    rec = rel[safe_ca].mean() if safe_ca.sum() else 0.0
    if fpr <= 0.051 and rec > best_recall:
        best_recall, best_t = rec, float(t)
print(f"tuned threshold t={best_t} (cal recall={best_recall:.4f})")

unsafe_de = (DB[de_m] > 0) | (DI[de_m] >= 0)
safe_de = L[de_m] == 1
rel = s_de >= best_t
fpr_de = rel[unsafe_de].mean()
rec_de = rel[safe_de].mean()
bal_de = balanced_accuracy_score(L[de_m], rel.astype(int))
auroc_de = roc_auc_score(L[de_m], s_de)
print(f"DEV: recall={rec_de:.4f} unsafe_fpr={fpr_de:.4f} bal_acc={bal_de:.4f} auroc={auroc_de:.4f}")
bars = {
    "unsafe_fpr_le_005": fpr_de <= 0.05,
    "release_recall_ge_050": rec_de >= 0.50,
    "auroc_ge_075": auroc_de >= 0.75,
    "balanced_acc_ge_070": bal_de >= 0.70,
}
print("bars:", json.dumps({k: bool(v) for k, v in bars.items()}))
print("DECISION:", "PASS" if all(bars.values()) else "FAIL")
