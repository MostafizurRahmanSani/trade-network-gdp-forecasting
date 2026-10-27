"""Conformal prediction with a calibration period that is separate from checkpoint and model selection.

Protocol ('cp' split): training inputs t <= 2011 (targets <= 2012) | early stopping and selection on targets 2013-14 | calibration on targets 2015-17 | test on targets 2018-24.
Point forecasts: five-seed ensemble mean for the neural models; tabular and ARIMA models are retrained on the cp split (hyper-parameters chosen on 2013-14).
Methods: split CP, volatility-normalised CP, adaptive conformal inference (ACI), rolling CP, expanding-window CP and group-conditional (Mondrian) CP, at 90% and 95%,
with gradient-boosting and linear quantile regression, a seed-spread interval and two naive forecasts as comparison. Also saves the tabular CP-protocol forecasts.
Outputs: outputs/conformal_results.csv/.md, conformal_seed_variability.csv.
"""
import os, sys, glob
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, selection
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
G = P.pivot(index="iso3", columns="year", values="g").loc[order]; Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
TS = {"train": range(2001, 2012), "val": range(2012, 2014), "cal": range(2014, 2017), "test": range(2017, 2024)}
YR = {k: np.repeat([t + 1 for t in v], N) for k, v in TS.items()}
Y = {k: np.concatenate([G[t + 1].values for t in v]) for k, v in TS.items()}
FE = ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi", "partner_ex_g", "partner_im_g"]
lo, hi = np.percentile(np.concatenate([G[t + 1].values for t in TS["train"]]), [1, 99])


import tabwin as W


def X_of(split):
    return np.concatenate([W.window(Pw, t) for t in TS[split]])


Xs = {k: X_of(k) for k in TS}
a_, b_ = np.nanpercentile(Xs["train"], 1, 0), np.nanpercentile(Xs["train"], 99, 0); c_ = np.clip(Xs["train"], a_, b_); mu_, sd_ = c_.mean(0), c_.std(0) + 1e-9
Z = {k: np.nan_to_num((np.clip(v, a_, b_) - mu_) / sd_) for k, v in Xs.items()}
ytr = np.clip(Y["train"], lo, hi)



def cp_config(m):
    """Configuration chosen on the CP early-stopping targets 2013-14 only (val_MAE of the @cp runs)."""
    import json
    cand = []
    for h in (16, 32):
        for wd in ("0", "1e-3"):
            fs = [RUNS / f"{m}@cp_base_h{h}_wd{wd}_s{s}.json" for s in (42, 1, 2, 3, 4)]
            if all(f.exists() for f in fs):
                v = float(np.mean([json.load(open(f))["val_MAE"] for f in fs])); cand.append((round(v / 1e-4), h, wd == "0", v, (m, "base", h, wd)))
    cand.sort(key=lambda r: r[:3]); print("CP-protocol selection", m, [(c[4][2], c[4][3], round(c[3], 5)) for c in cand], flush=True)
    return cand[0][4]

pts = {}
for m in ("STGNN", "LSTM", "GRU"):
    rn, cf, ch, cw = cp_config(m)
    fs = [f for f in sorted(glob.glob(str(RUNS / f"{f'{rn}@cp'}_{cf}_h{ch}_wd{cw}_s*.npz"))) if f.rsplit("_s", 1)[1][:-4].isdigit()]
    D = [np.load(f) for f in fs]
    if len(D) < 5: print("missing seeds for", m, len(D)); continue
    pts[m + " (5-seed mean)"] = {"cal": np.mean([d["cal_pred"] for d in D], 0), "test": np.mean([d["test_pred"] for d in D], 0)}
    pts["_ens_" + m] = {"sd": np.std([d["test_pred"] for d in D], 0)}
import itertools
bestg = None
for lr, dep, leaf, iters in itertools.product((0.03, 0.05, 0.1), (2, 3, 4, 6), (20, 50, 100), (100, 200, 400)):
    g = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=lr, max_depth=dep, min_samples_leaf=leaf, max_iter=iters, random_state=0).fit(Z["train"], ytr)
    e = np.abs(g.predict(Z["val"]) - Y["val"]).mean()
    if bestg is None or e < bestg[0]: bestg = (e, g)
pts["GBM"] = {"cal": bestg[1].predict(Z["cal"]), "test": bestg[1].predict(Z["test"])}
_, rfp = W.rf_seeds(Z["train"], ytr, Z["val"], Y["val"], list(itertools.product((3, 4, 6, 8, 12), (10, 50, 100), (0.3, 0.6, 1.0))), {"cal": Z["cal"], "test": Z["test"]})
pts["RandomForest"] = {"cal": rfp["cal"].mean(0), "test": rfp["test"].mean(0)}
A = lambda k: np.column_stack([np.concatenate([G[t].values for t in TS[k]]), np.ones(len(Y[k]))])
b = np.linalg.lstsq(np.nan_to_num(A("train")), ytr, rcond=None)[0]; pts["AR(1) pooled"] = {"cal": np.nan_to_num(A("cal")) @ b, "test": np.nan_to_num(A("test")) @ b}
for _k, _f in (("GBM", "CPP2_GBM"), ("RandomForest", "CPP2_RandomForest"), ("AR(1) pooled", "CPP2_AR1_pooled")):
    np.savez(OUT / "baselines" / f"{_f}.npz", cal_pred=pts[_k]["cal"], test_pred=pts[_k]["test"], cal_true=Y["cal"], test_true=Y["test"])

ac = OUT / "baselines" / "ARIMA_cp.npz"
if ac.exists():
    ar = np.load(ac); pts["ARIMA rolling"] = {"cal": ar["cal_pred"], "test": ar["test_pred"]}
else:
    ar = np.load(OUT / "baselines" / "ARIMA_rolling.npz"); pts["ARIMA rolling"] = {"cal": ar["val_pred"], "test": ar["test_pred"]}

acd = OUT / "baselines" / "ARIMA_cp_drift.npz"
if acd.exists():
    ard = np.load(acd); pts["ARIMA rolling, drift"] = {"cal": ard["cal_pred"], "test": ard["test_pred"]}


def conf_q(scores, alpha):
    n = len(scores); level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, method="higher")


size = P[P.year == 2017].set_index("iso3").loc[order, "loggdp"].values
vol = G.loc[:, 1996:2017].std(axis=1).values
inc = (P[P.year == 2017].set_index("iso3").loc[order, "loggdp"] - P[P.year == 2017].set_index("iso3").loc[order, "logpop"]).values
tert = lambda x: np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]))
groups = {"size": tert(size), "volatility": tert(vol), "income(GDPpc proxy)": tert(inc)}
gnames = {"size": ["small", "medium", "large"], "volatility": ["low", "medium", "high"], "income(GDPpc proxy)": ["low", "middle", "high"]}
sigma = {k: np.concatenate([np.maximum(G.loc[:, 1996:t].std(axis=1).values, 0.01) for t in TS[k]]) for k in ("cal", "test")}


def winkler(lo_, hi_, y, alpha):
    return (hi_ - lo_) + (2 / alpha) * (lo_ - y) * (y < lo_) + (2 / alpha) * (y - hi_) * (y > hi_)


def intervals(name, method, alpha):
    pc, pt = pts[name]["cal"], pts[name]["test"]; sc = np.abs(Y["cal"] - pc); yt = Y["test"]
    if method == "split-CP":
        q = conf_q(sc, alpha); return pt - q, pt + q, q
    if method == "normalised-CP":
        q = conf_q(sc / sigma["cal"], alpha); h = q * sigma["test"]; return pt - h, pt + h, q
    years = TS["test"]
    lo_, hi_ = np.zeros_like(pt), np.zeros_like(pt); qs = []
    pool = [sc]; recent = [sc[i * N:(i + 1) * N] for i in range(3)]; a_t = alpha
    for i, t in enumerate(years):
        sl = slice(i * N, (i + 1) * N); sc_t = np.abs(yt[sl] - pt[sl])
        if method == "ACI":
            q = np.quantile(np.concatenate(pool), 1 - a_t, method="higher")
        else:
            q = conf_q(np.concatenate(recent[-3:]), alpha)
        lo_[sl], hi_[sl] = pt[sl] - q, pt[sl] + q; qs.append(q)
        cov_t = np.mean(sc_t <= q); pool.append(sc_t); recent.append(sc_t)
        a_t = np.clip(a_t + 0.05 * (alpha - (1 - cov_t)), 0.001, 0.999)
    return lo_, hi_, float(np.mean(qs))


def evaluate(lo_, hi_, alpha):
    y = Y["test"]; cov = (y >= lo_) & (y <= hi_); r = {"coverage": cov.mean(), "width": float(np.mean(hi_ - lo_)), "winkler": float(winkler(lo_, hi_, y, alpha).mean())}
    w_ = winkler(lo_, hi_, y, alpha)
    for t in range(2018, 2025): r[f"cov{t}"] = cov[YR["test"] == t].mean(); r[f"wink{t}"] = w_[YR["test"] == t].mean()
    for gn, lab in groups.items():
        lab_t = np.tile(lab, len(TS["test"]))
        for k, nm in enumerate(gnames[gn]): r[f"cov[{gn}:{nm}]"] = cov[lab_t == k].mean(); r[f"wink[{gn}:{nm}]"] = w_[lab_t == k].mean()
    return r


rows = []
names = [k for k in pts if not k.startswith("_")]
for alpha in (0.10, 0.05):
    for nm in names:
        for meth in ("split-CP", "normalised-CP", "ACI", "rolling-CP"):
            l, h, q = intervals(nm, meth, alpha); r = evaluate(l, h, alpha); r.update(model=nm, method=meth, alpha=alpha, q=q, mae=float(np.abs(Y["test"] - pts[nm]["test"]).mean())); rows.append(r)
    for lvl, nm in ((alpha / 2, "qr"),):
        for kind in ("GBM-quantile", "Linear-quantile"):
            if kind == "GBM-quantile":
                ql = HistGradientBoostingRegressor(loss="quantile", quantile=alpha / 2, learning_rate=0.05, max_depth=3, min_samples_leaf=30, max_iter=150, random_state=0).fit(Z["train"], ytr)
                qh = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha / 2, learning_rate=0.05, max_depth=3, min_samples_leaf=30, max_iter=150, random_state=0).fit(Z["train"], ytr)
            else:
                ql = QuantileRegressor(quantile=alpha / 2, alpha=1e-3, solver="highs").fit(Z["train"], ytr); qh = QuantileRegressor(quantile=1 - alpha / 2, alpha=1e-3, solver="highs").fit(Z["train"], ytr)
            l, h = np.minimum(ql.predict(Z["test"]), qh.predict(Z["test"])), np.maximum(ql.predict(Z["test"]), qh.predict(Z["test"]))
            r = evaluate(l, h, alpha); r.update(model=kind, method="native quantile", alpha=alpha, q=float(np.mean(h - l) / 2), mae=float(np.abs(Y["test"] - (l + h) / 2).mean())); rows.append(r)
    if "_ens_STGNN" in pts:
        sd = pts["_ens_STGNN"]["sd"]; z = stats.norm.ppf(1 - alpha / 2); pt = pts["STGNN (5-seed mean)"]["test"]
        r = evaluate(pt - z * sd, pt + z * sd, alpha); r.update(model="STGNN deep ensemble", method="Gaussian (seed spread)", alpha=alpha, q=float(np.mean(z * sd)), mae=float(np.abs(Y["test"] - pt).mean())); rows.append(r)

R = pd.DataFrame(rows); R.to_csv(OUT / "conformal_results.csv", index=False)
md = ["# Conformal prediction (calibration targets 2015-17 separate from early stopping 2013-14; test 2018-24)", ""]
for alpha in (0.10, 0.05):
    sub = R[(R.alpha == alpha)]
    md += [f"## Nominal coverage {1 - alpha:.0%}", "", "```", sub[["model", "method", "coverage", "width", "winkler", "q", "mae"] + [f"cov{t}" for t in range(2018, 2025)]].round(3).to_string(index=False), "```", "",
           "Coverage by country group (split-CP):", "", "```", sub[sub.method == "split-CP"][["model"] + [c for c in R.columns if c.startswith("cov[")]].round(3).to_string(index=False), "```", ""]
md += ["## Winkler score by test year and by country group (split-CP, 90%)", "", "```",
       R[(R.alpha == 0.10) & (R.method == "split-CP")][["model"] + [f"wink{t}" for t in range(2018, 2025)] + [c for c in R.columns if c.startswith("wink[")]].round(3).to_string(index=False), "```", ""]
sd_rows = []
for m in ("STGNN", "LSTM", "GRU"):
    rn, cf, ch, cw = cp_config(m)
    fs = [f for f in sorted(glob.glob(str(RUNS / f"{f'{rn}@cp'}_{cf}_h{ch}_wd{cw}_s*.npz"))) if f.rsplit("_s", 1)[1][:-4].isdigit()]
    for i, f in enumerate(fs):
        d_ = np.load(f); pts["_seed"] = {"cal": d_["cal_pred"], "test": d_["test_pred"]}
        for alpha in (0.10, 0.05):
            for meth in ("split-CP", "normalised-CP", "ACI", "rolling-CP"):
                l, h, q = intervals("_seed", meth, alpha); r = evaluate(l, h, alpha)
                sd_rows.append(dict(model=m, method=meth, alpha=alpha, seed_idx=i, coverage=r["coverage"], width=r["width"], winkler=r["winkler"]))
del pts["_seed"]
SDR = pd.DataFrame(sd_rows); SDR.to_csv(OUT / "conformal_seed_variability.csv", index=False)
SD = SDR.groupby(["alpha", "method", "model"])[["coverage", "width", "winkler"]].agg(["mean", "std"]).round(4)
md += ["## Seed variability of the interval metrics (each of the 5 seeds calibrated separately; mean and sd over seeds; all methods, both levels)", "", "```", SD.to_string(), "```", ""]
s = R[(R.alpha == 0.10) & (R.method == "split-CP")]
rho = stats.spearmanr(s.mae, s.winkler)
md += [f"Spearman correlation between MAE and Winkler score across models (split-CP, 90%): {rho.correlation:.2f} (p={rho.pvalue:.3f}); with constant-width intervals this alignment is partly mechanical.", "",
       f"Effective calibration size: {len(Y['cal'])} residuals come from 3 calendar years; residuals within a year share global shocks, so the effective number of independent calibration points is closer to 3 than to {len(Y['cal'])}.", "",
       "Split-CP validity requires exchangeability between calibration and test residuals, which the 2020-21 shock violates; adaptive and rolling variants are reported for this reason."]
(OUT / "conformal_results.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[:30]))
