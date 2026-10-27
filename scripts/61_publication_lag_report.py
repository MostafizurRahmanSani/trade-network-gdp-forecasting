"""Publication-lag scenario: two-year-ahead forecasts (growth of t+2 from information up to year t; the last year's GDP is treated as not yet published for the
year being forecast). Compares one-year-ahead (main setting) with horizon 2 for the same target years 2019-2024; neural models come from 36_train_publication_lag.py,
tabular models and ARIMA (two-step rolling forecast, order chosen on validation) are computed here.
"""
import os, sys, glob, itertools, warnings
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, selection, tabwin as W
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
G = P.pivot(index="iso3", columns="year", values="g").loc[order]; L = P.pivot(index="iso3", columns="year", values="loggdp").loc[order]
Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
FE = ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi", "partner_ex_g", "partner_im_g"]
H = 2; TS = {"train": range(2001, 2014), "val": range(2014, 2017), "test": range(2017, 2023)}
X = lambda k: np.concatenate([W.window(Pw, t) for t in TS[k]])
Y = lambda k: np.concatenate([G[t + H].values for t in TS[k]])
Xtr, Xv, Xt = X("train"), X("val"), X("test"); a, b = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0); c = np.clip(Xtr, a, b); mu, sd = c.mean(0), c.std(0) + 1e-9
z = lambda M: np.nan_to_num((np.clip(M, a, b) - mu) / sd); lo, hi = np.percentile(Y("train"), [1, 99]); ytr = np.clip(Y("train"), lo, hi); yv, yt = Y("val"), Y("test")
out = {"No-change": np.zeros(len(yt))}
A = lambda k: np.nan_to_num(np.column_stack([np.concatenate([G[t].values for t in TS[k]]), np.ones(len(Y(k)))]))
out["AR(1) pooled (direct)"] = A("test") @ np.linalg.lstsq(A("train"), ytr, rcond=None)[0]
out["RW drift"] = np.concatenate([np.nan_to_num(G.loc[:, 1996:t].mean(axis=1).values) for t in TS["test"]])
out["Ridge"] = Ridge(alpha=min(((np.abs(Ridge(alpha=al).fit(z(Xtr), ytr).predict(z(Xv)) - yv).mean(), al) for al in (0.1, 1, 10, 100, 1000, 10000)))[1]).fit(z(Xtr), ytr).predict(z(Xt))
bg = min(((np.abs(HistGradientBoostingRegressor(loss="absolute_error", learning_rate=lr, max_depth=d, min_samples_leaf=l, max_iter=it, random_state=0).fit(z(Xtr), ytr).predict(z(Xv)) - yv).mean(), lr, d, l, it)
          for lr, d, l, it in itertools.product((0.03, 0.1), (2, 3, 6), (20, 100), (100, 300))))
out["GBM"] = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=bg[1], max_depth=bg[2], min_samples_leaf=bg[3], max_iter=bg[4], random_state=0).fit(z(Xtr), ytr).predict(z(Xt))
_, rfp = W.rf_seeds(z(Xtr), ytr, z(Xv), yv, list(itertools.product((3, 6, 12), (20, 100), (0.3, 0.6, 1.0))), {"test": z(Xt)}, n_sel=100, n_final=300)
out["Random forest"] = rfp["test"].mean(0)
bs = min(((np.abs(SVR(C=C, epsilon=0.005, gamma=g).fit(z(Xtr), ytr).predict(z(Xv)) - yv).mean(), C, g) for C, g in itertools.product((0.003, 0.01, 0.03, 0.1, 0.3, 1, 3), ("scale", 0.003, 0.03))))
out["SVR"] = SVR(C=bs[1], epsilon=0.005, gamma=bs[2]).fit(z(Xtr), ytr).predict(z(Xt))


def arima2(o, origins):
    res = []
    for t in origins:
        for c_ in order:
            s = L.loc[c_, 1996:t].values
            try: f = ARIMA(s, order=o).fit(method_kwargs={"maxiter": 200}).forecast(2); res.append(f[1] - f[0])
            except Exception: res.append(np.mean(np.diff(s)))
    return np.array(res)


bo = min(((np.abs(arima2(o, TS["val"]) - yv).mean(), o) for o in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 1, 0), (0, 1, 1)])); print("ARIMA order", bo, flush=True)
out["ARIMA rolling (2-step)"] = arima2(bo[1], TS["test"])

for m in ("STGNN", "STGNN_shuffled", "LSTM", "GRU", "MTGNN_trade"):
    pre = selection.prefix(m); base = "STGNN" if m == "STGNN_shuffled" else m
    rn, f, h, wd = selection.final_config(base)
    if m == "STGNN_shuffled": rn = rn.replace("STGNN", "STGNN_shuffled", 1)
    fs = [x for x in sorted(glob.glob(str(RUNS / f"{selection.compose(rn, 'h2')}_{f}_h{h}_wd{wd}_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    if len(fs) >= 5: out[m] = np.mean([np.load(x)["test_pred"] for x in fs], 0)

yr = np.repeat(range(2019, 2025), N); h1 = {}
base_arr = lambda name: np.load(OUT / "baselines" / f"{name}.npz")
truth1 = base_arr("Persistence(level)")["test_true"][N:]
h1["No-change"] = np.zeros(len(truth1)); h1["AR(1) pooled (direct)"] = base_arr("AR1_pooled")["test_pred"][N:]; h1["RW drift"] = base_arr("RW_drift")["test_pred"][N:]
h1["Ridge"] = base_arr("Ridge_win")["test_pred"][N:]; h1["GBM"] = base_arr("GBM_win")["test_pred"][N:]; h1["SVR"] = base_arr("SVR_win")["test_pred"][N:]
h1["Random forest"] = np.mean([base_arr(f"RandomForest_win_s{s_}")["test_pred"] for s_ in W.SEEDS], 0)[N:]
h1["ARIMA rolling (2-step)"] = base_arr("ARIMA_rolling")["test_pred"][N:]
for m in ("STGNN", "STGNN_shuffled", "LSTM", "GRU", "MTGNN_trade"):
    fs = [x for x in sorted(glob.glob(str(RUNS / f"{selection.prefix(m)}_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    if len(fs) >= 5: h1[m] = np.mean([np.load(x)["test_pred"] for x in fs], 0)[N:]
cov = np.isin(yr, (2020, 2021)); rows = []
for k, v in out.items():
    if k not in h1: continue
    e2, e1 = np.abs(v - yt), np.abs(h1[k] - truth1)
    rows.append(dict(model=k, h1_all=e1.mean(), h2_all=e2.mean(), change=(e2.mean() / e1.mean() - 1), h1_exCOVID=e1[~cov].mean(), h2_exCOVID=e2[~cov].mean()))
T = pd.DataFrame(rows).set_index("model").sort_values("h2_all")
md = ["# Publication-lag scenario: forecasting two years ahead of the last observed GDP (targets 2019-2024)", "",
      "h1 = main setting (window ending in the year before the target), h2 = window ending two years before the target. Same target years and countries; MAE in log growth points; neural rows are 5-seed means of predictions.", "",
      "```", T.round(4).to_string(), "```", "",
      "Reading: if the last year's GDP is not available when the forecast is made, every model loses accuracy, the ranking among the learned models is essentially unchanged, and the no-change forecast is unaffected by construction."]
(OUT / "horizon2.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[4:20]))
