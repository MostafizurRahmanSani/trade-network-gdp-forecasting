"""Post-hoc analyses on the saved predictions only (no model is trained or fitted): the point-accuracy table with Holm-adjusted Diebold-Mariano tests, exact
sign-flip randomisation tests, sign tests and the model confidence set; growth persistence and random-walk-with-drift references; bias, sign of forecasts and shrinkage
of the STGNN; ML regressors with and without trade-weighted partner growth; graph controls paired by seed; MAE without the samples affected by look-ahead imputation;
and the IMF October-forecast tests. Writes outputs/point_accuracy.csv and outputs/posthoc_tests.md.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, selection
OUT = ROOT / "outputs"; RUNS = OUT / "runs"; BASE = OUT / "baselines"
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
YEARS = list(range(2018, 2025)); yr = np.repeat(YEARS, N); cov = np.isin(yr, (2020, 2021))
G = P.pivot(index="iso3", columns="year", values="g").loc[order]
SEEDS = (42, 1, 2, 3, 4)


def seeds(prefix):
    fs = [RUNS / f"{prefix}_s{s}.npz" for s in SEEDS]
    if not all(f.exists() for f in fs): return None
    D = [np.load(f) for f in fs]
    return np.stack([d["val_pred"] for d in D]), np.stack([d["test_pred"] for d in D]), D[0]["val_true"], D[0]["test_true"]


FAM = {"STGNN": "STGNN", "STGCN": "STGCN", "STGAT": "STGAT", "DCRNN": "DCRNN", "GWN": "Graph WaveNet-style", "GNN": "GNN", "GCN": "GCN", "GAT": "GAT",
       "BonnigGAT": "Bonnig GAT", "BonnigGAT_edges": "Bonnig GAT, edge features", "GRU": "GRU", "LSTM": "LSTM", "RNN": "RNN", "BiLSTM": "BiLSTM",
       "Transformer": "Transformer", "TFT": "TFT-style", "MTGNN": "MTGNN-style", "MTGNN_trade": "MTGNN-style, trade graph"}
TAB = {"Persistence(level)": "No-change", "LastGrowth": "Growth persistence (last year's growth)", "RW_drift": "Random walk with drift", "CountryMeanGrowth": "Country mean growth",
       "AR1_pooled": "AR(1) pooled", "AR1+globalmean": "AR(1) pooled + global mean", "ARIMA_rolling_drift": "Rolling ARIMA with drift", "PanelAR2_FE": "Panel AR(2), fixed effects",
       "GVAR_style": "GVAR-type, pooled", "GVAR_country": "GVAR-type, country-specific", "DynamicFactor": "Dynamic factor model", "Ridge_win": "Ridge",
       "MedianReg_win": "Median regression", "GBM_win": "Gradient boosting", "SVR_win": "SVR"}
VARIANTS = {"STGNNX": "STGNN ablation reference (per-layer edge networks)", "STGNNX-l0": "STGNN variant: no message passing (0 layers)",
            "STGNNX-tlast": "STGNN variant: last input year only", "STGNNX-k1": "STGNN variant: skip from last-year features", "STGNNX-d2": "STGNN variant: dropout 0.2"}
INFO = {**{k: "N6+G" for k in FAM}, "GRU": "N6", "LSTM": "N6", "RNN": "N6", "BiLSTM": "N6", "Transformer": "N6", "TFT": "N6", "STGNNX-l0": "N6",
        "Persistence(level)": "-", "LastGrowth": "Y", "RW_drift": "Y", "CountryMeanGrowth": "Y", "AR1_pooled": "Y", "AR1+globalmean": "Y+W", "ARIMA_rolling_drift": "Y",
        "PanelAR2_FE": "Y", "GVAR_style": "Y+T", "GVAR_country": "Y+T", "DynamicFactor": "Y (panel)", "Ridge_win": "N6", "MedianReg_win": "N6", "GBM_win": "N6",
        "RandomForest_win": "N6", "SVR_win": "N6", "STGNNX": "N6+G", "STGNNX-tlast": "N6+G", "STGNNX-k1": "N6+G", "STGNNX-d2": "N6+G"}

pred = {}
for k in FAM:
    rn, f, h, wd = selection.final_config(k); r = seeds(f"{rn}_{f}_h{h}_wd{wd}")
    if r: pred[k] = (r[0], r[1]); truth_v, truth = r[2], r[3]
for k in VARIANTS:
    r = seeds(f"{k}_base_h16_wd1e-3")
    if r: pred[k] = (r[0], r[1])
for k in TAB:
    d = np.load(BASE / f"{k}.npz"); pred[k] = (d["val_pred"][None], d["test_pred"][None])
_rf = [np.load(BASE / f"RandomForest_win_s{s_}.npz") for s_ in SEEDS]; pred["RandomForest_win"] = (np.stack([d["val_pred"] for d in _rf]), np.stack([d["test_pred"] for d in _rf]))
TAB["RandomForest_win"] = "Random forest"
ens = {k: v[1].mean(0) for k, v in pred.items()}; ensv = {k: v[0].mean(0) for k, v in pred.items()}
NAME = {**FAM, **TAB, **VARIANTS}

scale = np.maximum(np.tile(np.abs(G.loc[:, 1997:2014].values).mean(axis=1), len(YEARS)), 0.005)
mae_nc, mae_rw = np.abs(truth).mean(), np.abs(ens["RW_drift"] - truth).mean()


def dm(a, b, mask=None):
    m = np.ones(len(truth), bool) if mask is None else mask
    ys = [t for t in YEARS if (m & (yr == t)).any()]
    d = np.array([(np.abs(ens[a].astype(float) - truth) - np.abs(ens[b].astype(float) - truth))[m & (yr == t)].mean() for t in ys], dtype=float); T_ = len(d)
    hln = d.mean() / (d.std(ddof=1) / np.sqrt(T_)) * np.sqrt((T_ - 1) / T_)
    p = 2 * stats.t.sf(abs(hln), T_ - 1); w = int((d < 0).sum())
    signs = np.array(np.meshgrid(*[[-1, 1]] * T_)).reshape(T_, -1).T
    p_rand = float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) * (1 - 1e-9)))
    p_sign = float(stats.binomtest(w, T_, 0.5).pvalue)
    return d.mean(), p, w, p_rand, p_sign


rows = []
for k, (pv, pt) in pred.items():
    e = np.abs(pt - truth); m = e.mean(0)
    rows.append(dict(key=k, model=NAME[k], info=INFO.get(k, ""), seeds=len(pt), val_MAE=np.abs(pv - truth_v).mean(1).mean(), test_MAE=e.mean(1).mean(),
                     sd=e.mean(1).std() if len(pt) > 1 else np.nan, ens_MAE=np.abs(pt.mean(0) - truth).mean(), RMSE=np.sqrt(((pt - truth) ** 2).mean(1)).mean(),
                     MASE=(m / scale).mean(), rel_nochange=m.mean() / mae_nc, rel_rwdrift=m.mean() / mae_rw, exCOVID=m[~cov].mean(), COVID=m[cov].mean()))
T = pd.DataFrame(rows).set_index("key")
for k in T.index:
    if k == "STGNN": continue
    T.loc[k, ["dm_diff", "dm_p", "W", "p_rand", "p_sign"]] = dm("STGNN", k)
MAIN = list(FAM) + list(TAB)
cmp_ = [k for k in MAIN if k != "STGNN"]
ps = T.loc[cmp_, "dm_p"].values; o = np.argsort(ps); holm = np.empty_like(ps); run = 0
for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); holm[i] = run
T.loc[cmp_, "dm_p_holm"] = holm
T = T.sort_values("test_MAE"); T.to_csv(OUT / "point_accuracy.csv")

def mcs(keys, B=10000, seed=0):
    L = np.array([[np.abs(ens[k] - truth)[yr == t].mean() for k in keys] for t in YEARS]); Tn = len(YEARS)
    rng = np.random.RandomState(seed); Lb = L[rng.randint(0, Tn, (B, Tn))]
    live = list(range(len(keys))); pv = {}; pmax = 0.0
    while len(live) > 1:
        Ll, Lbl = L[:, live], Lb[:, :, live]
        dbar = Ll.mean(0) - Ll.mean(); db = Lbl.mean(1) - Lbl.mean((1, 2))[:, None]
        sd_ = np.sqrt(((db - dbar) ** 2).mean(0)) + 1e-12; t = dbar / sd_
        p = float(np.mean(((db - dbar) / sd_).max(1) >= t.max())); pmax = max(pmax, p)
        w = live[int(np.argmax(t))]; pv[keys[w]] = pmax; live.remove(w)
    pv[keys[live[0]]] = 1.0
    return pv
mcs_p = mcs(MAIN)
T["mcs_p"] = [mcs_p.get(k, np.nan) for k in T.index]

md = ["# Post-hoc analyses on saved predictions (no model refitted)", "",
      "## A. Table 5 extended", "",
      "test_MAE = mean of per-seed MAE; ens_MAE = MAE of the seed-averaged forecast (used by all DM tests); rel_rwdrift = MAE / MAE of the random walk with drift; "
      "dm_p = year-clustered DM (HLN, t(6)) of the STGNN against the model, on seed-averaged forecasts; W = years (of 7) the STGNN is better; p_rand = exact sign-flip randomisation "
      "test of the mean yearly differential (128 sign patterns); p_sign = two-sided sign test on W; dm_p_holm = Holm adjustment over the 33 comparisons of the STGNN with the other models of Table 5; "
      "mcs_p = model-confidence-set p-value (T_max, i.i.d. bootstrap over the 7 test years, 10,000 draws) over the 34 models of Table 5 (variants excluded); a model is in the 90% MCS if mcs_p >= 0.10.",
      "Info: N6 = six base node features; +G = trade graph; tabular models of Table 5 receive the N6 window (36 inputs); Y = own growth history; "
      "+W = cross-country mean growth; +T = trade-weighted partner growth.", "", "```",
      T[["model", "info", "seeds", "val_MAE", "test_MAE", "sd", "ens_MAE", "RMSE", "MASE", "rel_nochange", "rel_rwdrift", "exCOVID", "COVID", "dm_diff", "dm_p", "W", "p_rand", "p_sign", "dm_p_holm", "mcs_p"]]
      .round({"val_MAE": 5, "test_MAE": 5, "sd": 5, "ens_MAE": 5, "RMSE": 4, "MASE": 3, "rel_nochange": 3, "rel_rwdrift": 3, "exCOVID": 4, "COVID": 4, "dm_diff": 5, "dm_p": 3, "p_rand": 3, "p_sign": 3, "dm_p_holm": 3, "mcs_p": 3}).to_string(), "```", ""]
inm = [NAME[k] for k in MAIN if mcs_p[k] >= 0.10]; out_ = [NAME[k] for k in MAIN if mcs_p[k] < 0.10]
md += [f"90% MCS contains {len(inm)} of {len(MAIN)} models; excluded: {', '.join(out_) if out_ else 'none'}.", ""]
rk = T.loc[MAIN].sort_values("val_MAE"); md += ["Ranking of the 34 models by validation MAE (top 10): " + ", ".join(f"{NAME[k]} {v:.5f}" for k, v in rk.val_MAE.head(10).items()), "",
     f"Rank of the STGNN by validation MAE among the 34 models: {int(list(rk.index).index('STGNN')) + 1}; by test MAE: {int(list(T.loc[MAIN].sort_values('test_MAE').index).index('STGNN')) + 1}.", ""]
vv = T.loc[["STGNNX", "STGNNX-l0", "STGNNX-tlast", "STGNNX-k1", "STGNNX-d2"]]
if "STGNNX-tlast" in T.index:
    rk2 = T.loc[MAIN + ["STGNNX-tlast"]].sort_values("test_MAE"); md += [f"Test-MAE rank of 'last input year only' if it replaced the STGNN among the 34 models: {list(rk2.drop('STGNN').index).index('STGNNX-tlast') + 1} of 34.", ""]

KEY = ["STGNN", "STGNNX-l0", "STGNNX-tlast", "RandomForest_win", "SVR_win", "GBM_win", "GRU", "LSTM", "AR1_pooled", "RW_drift"]
neg = truth < 0
md += ["## B. Bias, sign of forecasts and shrinkage (seed-averaged forecasts)", "", f"Negative outcomes in the test period: {int(neg.sum())} of {len(truth)} country-years "
       f"({', '.join(f'{t}: {int((neg & (yr == t)).sum())}' for t in YEARS)}).", "",
       "| Model | min forecast | 5th pct | 95th pct | negative forecasts | hit rate on negative outcomes | sign accuracy | " + " | ".join(f"bias {t}" for t in YEARS) + " | bias all |", "|---|" + "---|" * (7 + len(YEARS))]
for k in KEY:
    f_ = ens[k]; e = f_ - truth
    md.append(f"| {NAME[k]} | {f_.min():.4f} | {np.percentile(f_, 5):.4f} | {np.percentile(f_, 95):.4f} | {int((f_ < 0).sum())} | {np.mean(f_[neg] < 0):.3f} | {np.mean(np.sign(f_) == np.sign(truth)):.3f} | "
              + " | ".join(f"{e[yr == t].mean():+.4f}" for t in YEARS) + f" | {e.mean():+.4f} |")
md += ["", "Bias = mean of (forecast - outcome). Hit rate = share of negative outcomes with a negative forecast. Sign accuracy = share of country-years with the correct sign.", ""]
for c in ("VEN", "LBY", "SDN", "LBN") if "LBN" in order else ("VEN", "LBY", "SDN"):
    if c in order:
        i = order.index(c); es_ = ens["STGNN"]; o1 = ", ".join(f"{truth[j * N + i]:+.3f}" for j in range(7)); o2 = ", ".join(f"{es_[j * N + i]:+.3f}" for j in range(7))
        md.append(f"- {c}: outcomes {o1}; STGNN {o2}")
md.append("")

_f = selection.final_config("STGNN"); fx = f"{_f[1]}_h{_f[2]}_wd{_f[3]}"
real = seeds(f"STGNN_{fx}")
md += ["## C. Graph controls, paired by seed (real minus control; per-seed test MAE differences, mean and sd over the 5 seeds; paired t-test) and on seed-averaged forecasts", "",
       "| Control | per-seed diff mean | sd | paired t p | seed-averaged diff | DM p |", "|---|---|---|---|---|---|"]
for c in ("shuffled", "uniform", "shufw", "randedges", "static", "geo", "undirected", "reversed"):
    r = seeds(f"STGNN_{c}_{fx}")
    if r is None: continue
    dseed = np.abs(real[1] - truth).mean(1) - np.abs(r[1] - truth).mean(1); tt = stats.ttest_rel(np.abs(real[1] - truth).mean(1), np.abs(r[1] - truth).mean(1))
    ens[f"ctl_{c}"] = r[1].mean(0); dd = dm("STGNN", f"ctl_{c}")
    md.append(f"| {c} | {dseed.mean():+.5f} | {dseed.std(ddof=1):.5f} | {tt.pvalue:.3f} | {dd[0]:+.5f} | {dd[1]:.3f} |")
md += ["", "The year-clustered DM statistic divides a mean differential by its standard deviation; for two almost identical forecasts both are tiny, so differences of a few 1e-5 can be 'significant'.", ""]

md += ["## D. Top-k sensitivity recomputed on seed-averaged forecasts (same runs as topk_sensitivity.md, which averaged per-seed absolute errors)", "", "| k | diff (real - shuffled) | DM p | per-seed-error convention diff (topk_sensitivity.md) |", "|---|---|---|---|"]
for k, suf in ((5, "@k5"), (12, ""), (25, "@k25"), ("all", "@kall")):
    r1, r2 = seeds(f"STGNN{suf}_{fx}"), seeds(f"STGNN_shuffled{suf}_{fx}")
    if r1 is None or r2 is None: continue
    ens["_tk_r"], ens["_tk_s"] = r1[1].mean(0), r2[1].mean(0); dd = dm("_tk_r", "_tk_s")
    er, es = np.abs(r1[1] - truth).mean(0), np.abs(r2[1] - truth).mean(0)
    md.append(f"| {k} | {dd[0]:+.5f} | {dd[1]:.3f} | {np.mean([(er - es)[yr == t].mean() for t in YEARS]):+.5f} |")
S = pd.read_csv(OUT / "graph_density_full_network.csv"); md += ["", f"Edges per year in topk_sensitivity.md are means over 2016-2023 (k=12: {S[S.year.between(2016, 2023)].edges_k12.mean():.0f}; range over all years in graph_analysis.md).", ""]

r0 = seeds(f"STGNN_base_h{_f[2]}_wd0")
if r0:
    e0 = np.abs(r0[1] - truth).mean(1); v0 = np.abs(r0[0] - truth_v).mean(1)
    md += ["## E. STGNN without weight decay", "", f"Validation MAE {v0.mean():.5f} (seed sd {v0.std(ddof=1):.5f}); test MAE {e0.mean():.5f} (seed sd {e0.std(ddof=1):.5f}, population sd {e0.std():.5f}).", ""]

md += ["## F. Robustness variants of the ML regressors (not in Table 5): 11 node features (no partner growth) versus 13 features (with trade-weighted partner growth), each with four growth lags", "", "| Model | val 11 | val 13 | test 11 | test 13 | DM p (11 vs 13) | years 13 better |", "|---|---|---|---|---|---|---|"]
for b in ("Ridge", "MedianReg", "GBM"):
    a1, a2 = np.load(BASE / f"{b}_ext.npz"), np.load(BASE / f"{b}_partner.npz"); ens["_a"], ens["_b"] = a1["test_pred"], a2["test_pred"]; dd = dm("_b", "_a")
    md.append(f"| {b} | {np.abs(a1['val_pred'] - a1['val_true']).mean():.5f} | {np.abs(a2['val_pred'] - a2['val_true']).mean():.5f} | {np.abs(a1['test_pred'] - truth).mean():.5f} | {np.abs(a2['test_pred'] - truth).mean():.5f} | {dd[1]:.3f} | {dd[2]} |")
md += ["", "Random forest and SVR were run only with the 13-feature set (11_baselines_panel_gvar.py).", ""]

aff = pd.read_csv(OUT / "imputation_lookahead_samples.csv"); aff = aff[aff.split == "test"]
mask = np.ones(len(truth), bool)
for r in aff.itertuples(): mask[(r.target - 2018) * N + order.index(r.iso3)] = False
md += ["## G. Test MAE without the samples whose input window contains a future-dependent interpolated value", "", f"Samples removed: {int((~mask).sum())} of {len(truth)}.", "",
       "| Model | all | without affected samples |", "|---|---|---|"]
for k in ("STGNN", "RandomForest_win", "SVR_win", "GBM_win", "GAT", "STGCN", "LSTM", "GRU", "AR1_pooled", "RW_drift", "Persistence(level)"):
    md.append(f"| {NAME[k]} | {np.abs(ens[k] - truth).mean():.5f} | {np.abs(ens[k] - truth)[mask].mean():.5f} |")
md.append("")

try:
    import xlsx_reader as X
    rows_ = X.read_sheet(str(ROOT / "data" / "external" / "weo" / "weohistorical.xlsx"), "xl/worksheets/sheet2.xml")
    W = pd.DataFrame(rows_[1:], columns=rows_[0]).replace({".": np.nan, "": np.nan}); W["year"] = W["year"].astype(float)
    imf = {}
    for tag, col in (("IMF_fall", lambda Y: f"F{Y-1}ngdp_rpch"),):
        mat = np.full((len(YEARS), N), np.nan)
        for i, Y in enumerate(YEARS):
            c = col(Y)
            if c not in W.columns: continue
            sub = W[(W.year == Y) & W["ISOAlpha_3Code"].notna()].drop_duplicates("ISOAlpha_3Code").set_index("ISOAlpha_3Code")[c].astype(float)
            mat[i] = np.log1p(sub.reindex(order).values / 100)
        imf[tag] = mat.ravel(); ens[tag] = np.nan_to_num(mat.ravel())
    ok = np.isin(yr, [2018, 2019, 2020, 2021, 2022, 2023]) & ~np.isnan(imf["IMF_fall"])
    miss = sorted({order[j % N] for j in np.where(np.isin(yr, [2018, 2019, 2020, 2021, 2022, 2023]) & ~ok)[0]})
    md += ["## H. IMF October forecast: year-clustered DM tests on the country-years with an IMF forecast (6 years, HLN, t(5))", "", f"Country-years: {int(ok.sum())}. Economies without an IMF October forecast in at least one year: {', '.join(miss)}.", "",
           "| Comparison | mean loss diff | DM p | years first better (of 6) |", "|---|---|---|---|"]
    for a, b in (("STGNN", "IMF_fall"), ("RandomForest_win", "IMF_fall"), ("RW_drift", "IMF_fall"), ("STGNN", "RandomForest_win")):
        dd = dm(a, b, ok); md.append(f"| {NAME.get(a, a)} vs {b} | {dd[0]:+.5f} | {dd[1]:.3f} | {dd[2]} |")
    md.append("")
except Exception as ex:
    md += [f"IMF part skipped: {ex}", ""]

(OUT / "posthoc_tests.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
