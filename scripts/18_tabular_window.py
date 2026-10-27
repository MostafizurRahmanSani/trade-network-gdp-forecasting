"""Tabular models (ridge, median regression, gradient boosting, random forest, SVR) for the main comparison: the six base node features as a flattened
six-year window (6 features x years t-5..t = 36 inputs, exactly the information of the STGNN input window without the graph), with the same grids and
validation-only selection as 10_baselines_pooled.py / 11_baselines_panel_gvar.py. Outputs: outputs/baselines/<Model>_win.npz. Single thread.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import itertools, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
OUT = ROOT / "outputs" / "baselines"
P, iso = V.load(); order = iso.iso3.tolist()
Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
G = P.pivot(index="iso3", columns="year", values="g").loc[order]
BASE = ["g", "loggdp", "logpop", "dpop", "infl", "emp"]
TS = {"train": range(V.FIRST_T, V.TRAIN_END + 1), "val": range(V.TRAIN_END + 1, V.VAL_END + 1), "test": range(V.VAL_END + 1, V.LAST_T + 1)}


def mats(split):
    X = np.concatenate([np.column_stack([Pw[t - j][c].values for j in range(6) for c in BASE]) for t in TS[split]])
    y = np.concatenate([G[t + 1].values for t in TS[split]]); return X, y


Xtr, ytr = mats("train"); Xv, yv = mats("val"); Xt, yt = mats("test")
lo, hi = np.percentile(ytr, [1, 99]); yc = np.clip(ytr, lo, hi)
a, b = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0); c = np.clip(Xtr, a, b); mu, sd = c.mean(0), c.std(0) + 1e-9
z = lambda X: np.nan_to_num((np.clip(X, a, b) - mu) / sd)
res = {}
best = min(((np.abs(Ridge(alpha=al).fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), al) for al in (0.01, 0.1, 1, 10, 100, 1000, 10000)))
m = Ridge(alpha=best[1]).fit(z(Xtr), yc); res["Ridge_win"] = (m.predict(z(Xv)), m.predict(z(Xt))); print("ridge", best, flush=True)
qb = min(((np.abs(QuantileRegressor(quantile=0.5, alpha=al, solver="highs").fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), al) for al in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)))
q = QuantileRegressor(quantile=0.5, alpha=qb[1], solver="highs").fit(z(Xtr), yc); res["MedianReg_win"] = (q.predict(z(Xv)), q.predict(z(Xt))); print("median", qb, flush=True)
cand = []
for lr, depth, leaf, iters in itertools.product((0.03, 0.05, 0.1), (2, 3, 4, 6), (20, 50, 100), (100, 200, 400)):
    g = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=lr, max_depth=depth, min_samples_leaf=leaf, max_iter=iters, random_state=0).fit(Xtr, yc)
    cand.append((np.abs(g.predict(Xv) - yv).mean(), g))
g = min(cand, key=lambda x: x[0])[1]; res["GBM_win"] = (g.predict(Xv), g.predict(Xt)); print("gbm done", flush=True)
bf = min(((np.abs(RandomForestRegressor(n_estimators=150, max_depth=d, min_samples_leaf=l, max_features=mf, n_jobs=1, random_state=0).fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), d, l, mf)
          for d, l, mf in itertools.product((3, 4, 6, 8, 12), (10, 20, 50, 100), (0.3, 0.6, 1.0))))
rf = RandomForestRegressor(n_estimators=300, max_depth=bf[1], min_samples_leaf=bf[2], max_features=bf[3], n_jobs=1, random_state=0).fit(z(Xtr), yc)
res["RandomForest_win"] = (rf.predict(z(Xv)), rf.predict(z(Xt))); print("rf", bf, flush=True)
bs = min(((np.abs(SVR(C=C, epsilon=e, gamma=gm).fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), C, e, gm)
          for C, e, gm in itertools.product((0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30), (0.001, 0.005, 0.01), ("scale", 0.003, 0.03))))
sv = SVR(C=bs[1], epsilon=bs[2], gamma=bs[3]).fit(z(Xtr), yc); res["SVR_win"] = (sv.predict(z(Xv)), sv.predict(z(Xt))); print("svr", bs, flush=True)
for k, (vp, tp) in res.items():
    np.savez(OUT / f"{k}.npz", val_pred=vp, val_true=yv, test_pred=tp, test_true=yt)
    print(f"{k:20s} val {np.abs(vp - yv).mean():.4f} test {np.abs(tp - yt).mean():.4f}", flush=True)
