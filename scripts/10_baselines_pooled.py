"""Naive, statistical and machine-learning baselines on the study split (pooled across countries). Hyper-parameters are chosen on validation MAE only.
Predictions are saved in the same order as those of the neural models (window t ascending, then country order).
"""
import itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
OUT = ROOT / "outputs" / "baselines"; OUT.mkdir(parents=True, exist_ok=True)

P, iso = V.load(); order = iso.iso3.tolist()
Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
Gm = P.pivot(index="iso3", columns="year", values="g").loc[order]
FE = ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi", "partner_ex_g", "partner_im_g"]


def feats(t, cols):
    X = np.column_stack([Pw[t][c].values for c in cols] + [Gm[t - k].values for k in range(1, 5)])
    return X


def split_rows(split):
    ts = {"train": range(V.FIRST_T, V.TRAIN_END + 1), "val": range(V.TRAIN_END + 1, V.VAL_END + 1), "test": range(V.VAL_END + 1, V.LAST_T + 1)}[split]
    return list(ts)


def mats(split, cols):
    ts = split_rows(split)
    X = np.concatenate([feats(t, cols) for t in ts]); y = np.concatenate([Gm[t + 1].values for t in ts])
    return X, y


def main():
    res = {}
    lo, hi = np.percentile(np.concatenate([Gm[t + 1].values for t in split_rows("train")]), [1, 99])
    for cols_name, cols in [("ext", FE[:11]), ("partner", FE)]:
        Xtr, ytr = mats("train", cols); Xv, yv = mats("val", cols); Xt, yt = mats("test", cols)
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
        z = lambda X: np.nan_to_num((np.clip(X, np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0)) - mu) / sd)
        ytr_c = np.clip(ytr, lo, hi)
        best = min(((np.abs(Ridge(alpha=a).fit(z(Xtr), ytr_c).predict(z(Xv)) - yv).mean(), a) for a in (0.01, 0.1, 1, 10, 100, 1000, 10000)))
        m = Ridge(alpha=best[1]).fit(z(Xtr), ytr_c); res[f"Ridge_{cols_name}"] = (m.predict(z(Xv)), m.predict(z(Xt)), yv, yt)
        qb = min(((np.abs(QuantileRegressor(quantile=0.5, alpha=a, solver="highs").fit(z(Xtr), ytr_c).predict(z(Xv)) - yv).mean(), a) for a in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)))
        q = QuantileRegressor(quantile=0.5, alpha=qb[1], solver="highs").fit(z(Xtr), ytr_c); res[f"MedianReg_{cols_name}"] = (q.predict(z(Xv)), q.predict(z(Xt)), yv, yt)
        cand = []
        for lr, depth, leaf, iters in itertools.product((0.03, 0.05, 0.1), (2, 3, 4, 6), (20, 50, 100), (100, 200, 400)):
            g = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=lr, max_depth=depth, min_samples_leaf=leaf, max_iter=iters, random_state=0).fit(Xtr, ytr_c)
            cand.append((np.abs(g.predict(Xv) - yv).mean(), g))
        g = min(cand, key=lambda c: c[0])[1]; res[f"GBM_{cols_name}"] = (g.predict(Xv), g.predict(Xt), yv, yt)
    Xtr, ytr = mats("train", ["g", "gmean"]); Xv, yv = mats("val", ["g", "gmean"]); Xt, yt = mats("test", ["g", "gmean"])
    ytr_c = np.clip(ytr, lo, hi)
    res["Persistence(level)"] = (np.zeros_like(yv), np.zeros_like(yt), yv, yt)
    res["LastGrowth"] = (Xv[:, 0], Xt[:, 0], yv, yt)
    res["CountryMeanGrowth"] = tuple()
    cm = Gm.loc[:, 1996:V.TRAIN_END + 1].mean(1).values
    rep = lambda split: np.concatenate([cm for _ in split_rows(split)])
    res["CountryMeanGrowth"] = (rep("val"), rep("test"), yv, yt)
    for nm, cs in [("AR1_pooled", [0]), ("AR1+globalmean", [0, 1])]:
        A = np.column_stack([Xtr[:, cs], np.ones(len(Xtr))]); b = np.linalg.lstsq(A, ytr_c, rcond=None)[0]
        f = lambda X: np.column_stack([X[:, cs], np.ones(len(X))]) @ b
        res[nm] = (f(Xv), f(Xt), yv, yt)
    for k, (vp, tp, vt, tt) in res.items():
        np.savez(OUT / f"{k}.npz", val_pred=vp, val_true=vt, test_pred=tp, test_true=tt)
        print(f"{k:22s} val {np.abs(vp-vt).mean():.4f} test {np.abs(tp-tt).mean():.4f}")


if __name__ == "__main__":
    main()
