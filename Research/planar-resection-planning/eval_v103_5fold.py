"""5-fold scene CV over Train-512 to characterise recall@FPR<=5% robustness.

Each fold: train a HistGBM on 4/5 of scenes, tune the safety threshold on a
held-out calibration subset (within the fold) to maximise recall subject to
unsafe-FPR<=5%, then evaluate recall/FPR/AUROC/bal-acc on the remaining fold.

This tells us whether recall@FPR<=5% ~0.5 is consistently achievable or
whether it is scene-split dependent (my fixed cal/dev split showed 0.16 vs
0.55 for the same model).
"""
import json, random, sys
from pathlib import Path
import numpy as np
SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import norm

DATA = Path("results/clinical_window_v10_2/pilot_v103")
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
scenes = sorted({a["scenario_id"] for a in audits})
rng = random.Random(20260811)
rng.shuffle(scenes)
folds = [scenes[i::5] for i in range(5)]
print(f"total scenes: {len(scenes)}, folds: {[len(f) for f in folds]}")

def mask_ids(ids):
    s = set(ids)
    return np.asarray([a["scenario_id"] in s for a in audits], bool)

results = []
for fi, test_ids in enumerate(folds):
    train_ids = [sid for j, f in enumerate(folds) if j != fi for sid in f]
    rng2 = random.Random(20260811 + fi)
    shuffled = list(train_ids); rng2.shuffle(shuffled)
    # within-fold: first 80% scenes = train, last 20% = calibration
    n_cal = max(10, len(shuffled) // 5)
    cal_ids = shuffled[:n_cal]
    tr_ids = shuffled[n_cal:]
    tr_m = mask_ids(tr_ids); ca_m = mask_ids(cal_ids); te_m = mask_ids(test_ids)
    sc = StandardScaler().fit(F[tr_m])
    Xtr, Xca, Xte = sc.transform(F[tr_m]), sc.transform(F[ca_m]), sc.transform(F[te_m])
    preds = {}
    for col in (0, 1):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4,
                                          random_state=0, validation_fraction=0.15, early_stopping=True)
        m.fit(Xtr, R[tr_m, col])
        preds[col] = (m.predict(Xca), m.predict(Xte))
    sb = float(np.std(preds[0][0] - R[ca_m, 0])); si = float(np.std(preds[1][0] - R[ca_m, 1]))
    def score(b, i): return norm.cdf(-b / max(sb, 1e-6)) * norm.cdf(-i / max(si, 1e-6))
    s_ca = score(preds[0][0], preds[1][0]); s_te = score(preds[0][1], preds[1][1])
    unsafe_ca = (DB[ca_m] > 0) | (DI[ca_m] >= 0); safe_ca = L[ca_m] == 1
    best_t, best_rec = None, -1.0
    for t in np.round(np.arange(0.0, 1.0, 0.005), 3):
        rel = s_ca >= t
        fpr = rel[unsafe_ca].mean() if unsafe_ca.sum() else 0.0
        rec = rel[safe_ca].mean() if safe_ca.sum() else 0.0
        if fpr <= 0.051 and rec > best_rec:
            best_rec, best_t = rec, float(t)
    if best_t is None:
        best_t = 0.5
    unsafe_te = (DB[te_m] > 0) | (DI[te_m] >= 0); safe_te = L[te_m] == 1
    rel = s_te >= best_t
    rec = rel[safe_te].mean() if safe_te.sum() else 0.0
    fpr = rel[unsafe_te].mean() if unsafe_te.sum() else 0.0
    bal = balanced_accuracy_score(L[te_m], rel.astype(int))
    auroc = roc_auc_score(L[te_m], s_te)
    results.append({"fold": fi, "t": best_t, "recall": float(rec), "fpr": float(fpr),
                    "bal_acc": float(bal), "auroc": float(auroc),
                    "n_test": int(len(te_m)), "n_safe": int(safe_te.sum())})
    print(f"fold {fi}: t={best_t:.3f} recall={rec:.4f} fpr={fpr:.4f} bal_acc={bal:.4f} auroc={auroc:.4f}")

recs = [r["recall"] for r in results]
print(f"\nrecall@FPR<=5% across folds: {['%.3f'%r for r in recs]}")
print(f"mean recall={np.mean(recs):.4f} median={np.median(recs):.4f} min={min(recs):.4f} max={max(recs):.4f}")
print(f"AUROC mean={np.mean([r['auroc'] for r in results]):.4f} bal_acc mean={np.mean([r['bal_acc'] for r in results]):.4f}")
print(f"folds passing recall>=0.50: {sum(1 for r in recs if r>=0.50)}/5")
print(f"folds passing all 4 bars: {sum(1 for r in results if r['fpr']<=0.05 and r['recall']>=0.5 and r['auroc']>=0.75 and r['bal_acc']>=0.70)}/5")
