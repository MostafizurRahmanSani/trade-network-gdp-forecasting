"""Classical multi-country baselines (single thread, same row order as the other baselines): random walk with country drift, random forest, SVR,
panel AR(2) with country fixed effects, and a GVAR-type regression (own lags plus trade-weighted partner growth).
Hyper-parameters are chosen on validation MAE only; training targets are winsorised as for the other baselines.
"""
import os, sys
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
G = P.pivot(index="iso3", columns="year", values="g").loc[order]
OUT = ROOT / "outputs" / "baselines"
FE = ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi", "partner_ex_g", "partner_im_g"]
TS = {"train": list(range(V.FIRST_T, V.TRAIN_END + 1)), "val": list(range(V.TRAIN_END + 1, V.VAL_END + 1)), "test": list(range(V.VAL_END + 1, V.LAST_T + 1))}
lo, hi = np.percentile(np.concatenate([G[t + 1].values for t in TS["train"]]), [1, 99])


def X_of(split, cols, lags=4):
    X = np.concatenate([np.column_stack([Pw[t][c].values for c in cols] + [G[t - k].values for k in range(1, lags + 1)]) for t in TS[split]])
    y = np.concatenate([G[t + 1].values for t in TS[split]]); return X, y


def zfun(Xtr):
    a, b = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0); c = np.clip(Xtr, a, b); mu, sd = c.mean(0), c.std(0) + 1e-9
    return lambda X: np.nan_to_num((np.clip(X, a, b) - mu) / sd)


res = {}
for nm in ("val", "test"):
    pass
drift = lambda split: np.concatenate([np.nan_to_num(G.loc[:, 1996:t].mean(axis=1).values) for t in TS[split]])
_, yv = X_of("val", FE[:11]); _, yt = X_of("test", FE[:11])
res["RW_drift"] = (drift("val"), drift("test"), yv, yt)

Xtr, ytr = X_of("train", FE); Xv, yv = X_of("val", FE); Xt, yt = X_of("test", FE); z = zfun(Xtr); yc = np.clip(ytr, lo, hi)
import itertools
best = min(((np.abs(RandomForestRegressor(n_estimators=150, max_depth=d, min_samples_leaf=l, max_features=mf, n_jobs=1, random_state=0).fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), d, l, mf)
            for d, l, mf in itertools.product((3, 4, 6, 8, 12), (10, 20, 50, 100), (0.3, 0.6, 1.0))))
rf = RandomForestRegressor(n_estimators=300, max_depth=best[1], min_samples_leaf=best[2], max_features=best[3], n_jobs=1, random_state=0).fit(z(Xtr), yc)
res["RandomForest"] = (rf.predict(z(Xv)), rf.predict(z(Xt)), yv, yt); print("RF", best, flush=True)
bs = min(((np.abs(SVR(C=C, epsilon=e, gamma=g).fit(z(Xtr), yc).predict(z(Xv)) - yv).mean(), C, e, g) for C, e, g in itertools.product((0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30), (0.001, 0.005, 0.01), ("scale", 0.003, 0.03))))
sv = SVR(C=bs[1], epsilon=bs[2], gamma=bs[3]).fit(z(Xtr), yc); res["SVR"] = (sv.predict(z(Xv)), sv.predict(z(Xt)), yv, yt); print("SVR", bs, flush=True)

def ols_rows(split, kind):
    rows = []
    for t in TS[split]:
        cols = [G[t].values, G[t - 1].values]
        if kind == "gvar": cols += [Pw[t]["partner_ex_g"].values, Pw[t]["partner_im_g"].values]
        rows.append(np.column_stack(cols + ([np.eye(N)] if kind == "panel" else [np.ones((N, 1))])))
    return np.nan_to_num(np.concatenate(rows))
for kind, nm in (("panel", "PanelAR2_FE"), ("gvar", "GVAR_style")):
    A = ols_rows("train", kind); b = np.linalg.lstsq(A, yc, rcond=None)[0]
    res[nm] = (ols_rows("val", kind) @ b, ols_rows("test", kind) @ b, yv, yt)

yr = np.repeat(range(2018, 2025), N); cov = np.isin(yr, (2020, 2021))
for k, (vp, tp, vt, tt) in res.items():
    np.savez(OUT / f"{k}.npz", val_pred=vp, val_true=vt, test_pred=tp, test_true=tt)
    e = np.abs(tp - tt); print(f"{k:14s} val {np.abs(vp - vt).mean():.4f} test {e.mean():.4f} exCOVID {e[~cov].mean():.4f} COVID {e[cov].mean():.4f}", flush=True)
