"""Country-specific GVAR-type model: for every country i an ARX* equation estimated by ridge regression on the training years,
    g_i(t+1) = a_i + b_i g_i(t) + c_i g*_ex,i(t) + d_i g*_im,i(t),
where g*_ex / g*_im are the trade-weighted growth rates of the country's export destinations / import sources (weights of the previous year).
The ridge penalty (common to all countries) is chosen on validation MAE.
"""
import os, sys
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
G = P.pivot(index="iso3", columns="year", values="g").loc[order]; Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
TS = {"train": range(V.FIRST_T, V.TRAIN_END + 1), "val": range(V.TRAIN_END + 1, V.VAL_END + 1), "test": range(V.VAL_END + 1, V.LAST_T + 1)}
lo, hi = np.percentile(np.concatenate([G[t + 1].values for t in TS["train"]]), [1, 99])
Xc = lambda t: np.column_stack([np.clip(G[t].values, lo, hi), Pw[t].partner_ex_g.values, Pw[t].partner_im_g.values])


def fit(lam):
    B = []
    for i in range(N):
        X = np.array([Xc(t)[i] for t in TS["train"]]); y = np.clip(np.array([G[t + 1].values[i] for t in TS["train"]]), lo, hi)
        mu, my = X.mean(0), y.mean(); Z = X - mu
        B.append((my, mu, np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(3) * Z.var(0).mean(), Z.T @ (y - my))))
    return B


def predict(B, split):
    out = []
    for t in TS[split]:
        X = Xc(t); out.append(np.array([B[i][0] + (X[i] - B[i][1]) @ B[i][2] for i in range(N)]))
    return np.concatenate(out)


yv = np.concatenate([G[t + 1].values for t in TS["val"]]); yt = np.concatenate([G[t + 1].values for t in TS["test"]])
best = min(((np.abs(predict(fit(l), "val") - yv).mean(), l) for l in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)))
B = fit(best[1]); vp, tp = predict(B, "val"), predict(B, "test")
np.savez(ROOT / "outputs" / "baselines" / "GVAR_country.npz", val_pred=vp, val_true=yv, test_pred=tp, test_true=yt)
yr = np.repeat(range(2018, 2025), N); cov = np.isin(yr, (2020, 2021)); e = np.abs(tp - yt)
print(f"GVAR_country lambda={best[1]} val {best[0]:.4f} test {e.mean():.4f} exCOVID {e[~cov].mean():.4f} COVID {e[cov].mean():.4f}")
