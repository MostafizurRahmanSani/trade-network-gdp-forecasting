"""Rolling-origin evaluation: three origins with retraining of every model.
  gfc : training inputs <= 2005 | selection 2006-07 | test targets 2009-10 (global financial crisis)
  mid : training inputs <= 2009 | selection 2010-11 | test targets 2013-16
  main: training inputs <= 2013 | selection 2014-16 | test targets 2018-24
Neural models keep the configuration selected on the main split; tabular baselines are re-tuned on each origin's own selection years.
"""
import os, sys, glob, itertools
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, selection, tabwin as W
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
G = P.pivot(index="iso3", columns="year", values="g").loc[order]; Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
FE = ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi", "partner_ex_g", "partner_im_g"]
FOLDS = {"gfc": dict(train_end=2005, val_end=2007, last_t=2009), "mid": dict(train_end=2009, val_end=2011, last_t=2015), "main": dict(train_end=2013, val_end=2016, last_t=2023)}
FAM = ["STGNN", "STGNN_shuffled", "STGCN", "STGAT", "DCRNN", "GWN", "GNN", "GCN", "GAT", "BonnigGAT", "BonnigGAT_edges", "GRU", "LSTM", "RNN", "BiLSTM", "Transformer", "TFT", "MTGNN", "MTGNN_trade"]
LAB = {"gfc": "2009-10 (crisis)", "mid": "2013-16", "main": "2018-24 (COVID)"}


def tab(fold):
    s = FOLDS[fold]; TS = {"train": range(2001, s["train_end"] + 1), "val": range(s["train_end"] + 1, s["val_end"] + 1), "test": range(s["val_end"] + 1, s["last_t"] + 1)}
    X = lambda k: np.concatenate([W.window(Pw, t) for t in TS[k]])
    Y = lambda k: np.concatenate([G[t + 1].values for t in TS[k]])
    Xtr, Xv, Xt = X("train"), X("val"), X("test"); a, b = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0); c = np.clip(Xtr, a, b); mu, sd = c.mean(0), c.std(0) + 1e-9
    z = lambda M: np.nan_to_num((np.clip(M, a, b) - mu) / sd); lo, hi = np.percentile(Y("train"), [1, 99]); ytr = np.clip(Y("train"), lo, hi); yv = Y("val")
    out = {"No-change": np.zeros(len(Y("test")))}
    A = lambda k: np.nan_to_num(np.column_stack([np.concatenate([G[t].values for t in TS[k]]), np.ones(len(Y(k)))]))
    out["AR(1) pooled"] = A("test") @ np.linalg.lstsq(A("train"), ytr, rcond=None)[0]
    out["RW drift"] = np.concatenate([np.nan_to_num(G.loc[:, 1996:t].mean(axis=1).values) for t in TS["test"]])
    out["Ridge"] = Ridge(alpha=min(((np.abs(Ridge(alpha=al).fit(z(Xtr), ytr).predict(z(Xv)) - yv).mean(), al) for al in (0.1, 1, 10, 100, 1000)))[1]).fit(z(Xtr), ytr).predict(z(Xt))
    best = min(((np.abs(HistGradientBoostingRegressor(loss="absolute_error", learning_rate=lr, max_depth=d, min_samples_leaf=l, max_iter=it, random_state=0).fit(z(Xtr), ytr).predict(z(Xv)) - yv).mean(), lr, d, l, it)
                for lr, d, l, it in itertools.product((0.03, 0.1), (3, 6), (20, 100), (100, 300))))
    out["GBM"] = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=best[1], max_depth=best[2], min_samples_leaf=best[3], max_iter=best[4], random_state=0).fit(z(Xtr), ytr).predict(z(Xt))
    _, rfp = W.rf_seeds(z(Xtr), ytr, z(Xv), yv, list(itertools.product((4, 8), (20, 100), (0.3, 1.0))), {"test": z(Xt)}, n_sel=100, n_final=200)
    out["Random forest"] = rfp["test"].mean(0)
    bs = min(((np.abs(SVR(C=C, epsilon=0.005, gamma=g).fit(z(Xtr), ytr).predict(z(Xv)) - yv).mean(), C, g) for C, g in itertools.product((0.003, 0.03, 0.3, 3), ("scale", 0.003))))
    out["SVR"] = SVR(C=bs[1], epsilon=0.005, gamma=bs[2]).fit(z(Xtr), ytr).predict(z(Xt))
    return out, Y("test"), np.repeat([t + 1 for t in TS["test"]], N)


def nn_pred(m, fold):
    base = "STGNN" if m == "STGNN_shuffled" else m
    cfg = selection.final_config(base)
    if cfg is None: return None
    name, f, h, wd = cfg
    if m == "STGNN_shuffled": name = name.replace("STGNN", "STGNN_shuffled", 1)
    pat = f"{name}_{f}_h{h}_wd{wd}" if fold == "main" else f"{selection.compose(name, fold)}_{f}_h{h}_wd{wd}"
    fs = [x for x in sorted(glob.glob(str(RUNS / (pat + "_s*.npz")))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    return np.stack([np.load(x)["test_pred"] for x in fs]) if len(fs) >= 5 else None


rows, detail, per_year = {}, {}, {}
for fold in ("gfc", "mid", "main"):
    out, y, yrs = tab(fold)
    for m in FAM:
        P_ = nn_pred(m, fold)
        if P_ is not None: out[m] = P_.mean(0); out[m + "|sd"] = np.abs(P_ - y).mean(1).std()
    for k, v in out.items():
        if k.endswith("|sd"): continue
        rows.setdefault(k, {})[fold] = np.abs(v - y).mean(); per_year.setdefault(fold, {})[k] = [np.abs(v - y)[yrs == t].mean() for t in sorted(set(yrs))]
    per_year[fold]["_years"] = sorted(set(yrs)); detail[fold] = len(y)
T = pd.DataFrame(rows).T[["gfc", "mid", "main"]]; T.columns = [LAB[c] for c in T.columns]
md = ["# Rolling-origin evaluation (test MAE, real GDP growth)", "", "Every neural family is retrained at each origin with its main-split configuration (5 seeds, seed-mean predictions); tabular models are re-tuned per origin.", "",
      "```", T.sort_values(LAB["main"]).round(4).to_string(), "```", "", "Per-year test MAE within each origin:", ""]
for fold in ("gfc", "mid", "main"):
    d = per_year[fold]; ys = d.pop("_years"); md += [f"**{LAB[fold]}**", "", "```", pd.DataFrame(d, index=ys).T.round(4).to_string(), "```", ""]
md += ["The 2009-10 origin has only 5 training windows, so neural models are data-starved there. Ranking correlation across origins:", ""]
rk = T.rank(); md += ["```", rk.corr(method="spearman").round(2).to_string(), "```"]
(OUT / "rolling_origins.md").write_text("\n".join(md), encoding="utf-8"); T.to_csv(OUT / "rolling_origins.csv"); print("\n".join(md[4:30]))
