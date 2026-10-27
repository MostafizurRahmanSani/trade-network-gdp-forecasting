"""Dynamic factor model baseline: principal-component factors of the (winsorised, country-demeaned) growth panel on the training years, VAR(1) dynamics
of the factors, forecast = country mean + loadings x predicted factor. The factor at the forecast origin is obtained by projecting the observed cross-section
of year t on the loadings (no future information). The number of factors is chosen on validation MAE.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
G = P.pivot(index="iso3", columns="year", values="g").loc[order]
lo, hi = np.nanpercentile(G.loc[:, 1996:V.TRAIN_END + 1].values, [1, 99]); Gc = G.clip(lo, hi)
yrs = list(range(1997, V.TRAIN_END + 2)); X = Gc.loc[:, yrs].values; mu = X.mean(1, keepdims=True); Xd = X - mu


def fit(K):
    U, S, Vt = np.linalg.svd(Xd, full_matrices=False); L = U[:, :K] * S[:K]; F = Vt[:K]
    A = (F[:, 1:] @ F[:, :-1].T) @ np.linalg.pinv(F[:, :-1] @ F[:, :-1].T)
    return L, A


def predict(L, A, ts):
    out = []
    for t in ts:
        x = Gc[t].values[:, None] - mu; f = np.linalg.lstsq(L, x, rcond=None)[0]
        out.append((mu + L @ (A @ f)).ravel())
    return np.concatenate(out)


VAL, TEST = list(range(V.TRAIN_END + 1, V.VAL_END + 1)), list(range(V.VAL_END + 1, V.LAST_T + 1))
yv = np.concatenate([G[t + 1].values for t in VAL]); yt = np.concatenate([G[t + 1].values for t in TEST])
best = None
for K in range(1, 11):
    L, A = fit(K); e = np.abs(predict(L, A, VAL) - yv).mean(); print("K", K, "val", round(e, 4))
    if best is None or e < best[0]: best = (e, K, L, A)
_, K, L, A = best; vp, tp = predict(L, A, VAL), predict(L, A, TEST)
np.savez(ROOT / "outputs" / "baselines" / "DynamicFactor.npz", val_pred=vp, val_true=yv, test_pred=tp, test_true=yt)
yr = np.repeat(range(2018, 2025), N); cov = np.isin(yr, (2020, 2021)); e = np.abs(tp - yt)
print(f"DynamicFactor K={K} val {best[0]:.4f} test {e.mean():.4f} exCOVID {e[~cov].mean():.4f} COVID {e[cov].mean():.4f}")
