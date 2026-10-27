"""Random forest and SVR with the 11 node features without trade-weighted partner growth and four growth lags, with the procedure and grids of
11_baselines_panel_gvar.py (which uses the 13-feature set). Gives the 11- versus 13-feature comparison.
Outputs: outputs/baselines/RandomForest_ext.npz, SVR_ext.npz.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import itertools, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
P, iso = V.load(); order = iso.iso3.tolist()
Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}; G = P.pivot(index="iso3", columns="year", values="g").loc[order]
OUT = ROOT / "outputs" / "baselines"
FE = ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi", "partner_ex_g", "partner_im_g"][:11]
TS = {"train": list(range(V.FIRST_T, V.TRAIN_END + 1)), "val": list(range(V.TRAIN_END + 1, V.VAL_END + 1)), "test": list(range(V.VAL_END + 1, V.LAST_T + 1))}
lo, hi = np.percentile(np.concatenate([G[t + 1].values for t in TS["train"]]), [1, 99])
X_of = lambda s: (np.concatenate([np.column_stack([Pw[t][c].values for c in FE] + [G[t - k].values for k in range(1, 5)]) for t in TS[s]]), np.concatenate([G[t + 1].values for t in TS[s]]))
Xtr, ytr = X_of("train"); Xv, yv = X_of("val"); Xt, yt = X_of("test")
a, b = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0); c = np.clip(Xtr, a, b); mu, sd = c.mean(0), c.std(0) + 1e-9
z = lambda X: np.nan_to_num((np.clip(X, a, b) - mu) / sd); yc = np.clip(ytr, lo, hi)
for fn in ("RandomForest_ext.npz", "SVR_ext.npz"): assert not (OUT / fn).exists()
best = min(((np.abs(RandomForestRegressor(n_estimators=150, max_depth=d, min_samples_leaf=l, max_features=mf, n_jobs=1, random_state=0).fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), d, l, mf)
            for d, l, mf in itertools.product((3, 4, 6, 8, 12), (10, 20, 50, 100), (0.3, 0.6, 1.0))))
rf = RandomForestRegressor(n_estimators=300, max_depth=best[1], min_samples_leaf=best[2], max_features=best[3], n_jobs=1, random_state=0).fit(z(Xtr), yc)
np.savez(OUT / "RandomForest_ext.npz", val_pred=rf.predict(z(Xv)), val_true=yv, test_pred=rf.predict(z(Xt)), test_true=yt); print("RF", best, flush=True)
bs = min(((np.abs(SVR(C=C, epsilon=e, gamma=g).fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), C, e, g) for C, e, g in itertools.product((0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30), (0.001, 0.005, 0.01), ("scale", 0.003, 0.03))))
sv = SVR(C=bs[1], epsilon=bs[2], gamma=bs[3]).fit(z(Xtr), yc)
np.savez(OUT / "SVR_ext.npz", val_pred=sv.predict(z(Xv)), val_true=yv, test_pred=sv.predict(z(Xt)), test_true=yt); print("SVR", bs, flush=True)
