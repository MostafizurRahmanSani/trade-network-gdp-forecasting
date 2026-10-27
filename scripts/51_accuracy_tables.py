"""Accuracy tables: seed statistics, MASE and relative MAE, ex-COVID and COVID splits, per-year and per-group errors, and year-clustered Diebold-Mariano
tests (Harvey-Leybourne-Newbold small-sample correction). For every neural family the validation-selected configuration is used (five seeds); the graph controls
use the STGNN configuration.
"""
import glob, json, re, sys
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


def load_family(name, cfg=None):
    """-> (config tag, val MAE mean, list of per-seed test preds, truth) for the family's FINAL configuration (validation-selected on the grid, or adopted by the
    extended tuning check; see selection.final_config), or for a fixed (feature set, hidden, weight decay) used by the graph controls."""
    if cfg is None:
        fc = selection.final_config(name)
        if fc is None: return None
        rn, f, h, wd = fc
    else:
        rn, (f, h, wd) = name, cfg
    fs = [RUNS / f"{rn}_{f}_h{h}_wd{wd}_s{s}.json" for s in (42, 1, 2, 3, 4)]
    fs = [x for x in fs if x.exists()]
    if len(fs) < 5: return None
    D = [np.load(x.with_suffix(".npz")) for x in fs]
    tag = f"{f},h{h},wd{wd}" + (f",lr{rn.split('#lr')[1]}" if "#lr" in rn else "")
    return tag, float(np.mean([json.load(open(x))["val_MAE"] for x in fs])), [d["test_pred"] for d in D], D[0]["test_true"]


_f = selection.final_config("STGNN"); FIX = (_f[1], str(_f[2]), _f[3]); LRSUF = "#lr" + _f[0].split("#lr")[1] if "#lr" in _f[0] else ""
FAMILIES = ["STGNN", "STGCN", "STGAT", "DCRNN", "GWN", "GNN", "GCN", "GAT", "BonnigGAT", "BonnigGAT_edges", "GRU", "LSTM", "RNN", "BiLSTM", "Transformer", "TFT", "MTGNN", "MTGNN_trade"]
CONTROLS = ["STGNN_shuffled", "STGNN_uniform", "STGNN_reversed", "STGNN_undirected", "STGNN_randedges", "STGNN_shufw", "STGNN_geo", "STGNN_static"]
models = {}
for f in FAMILIES:
    r = load_family(f)
    if r: models[f] = r
for c in CONTROLS:
    r = load_family(c + LRSUF, FIX)
    if r and len(r[2]) >= 5: models[c] = r
truth = next(iter(models.values()))[3]
tab = {}
for k in ("Persistence(level)", "RW_drift", "CountryMeanGrowth", "AR1_pooled", "AR1+globalmean", "ARIMA_rolling", "PanelAR2_FE", "GVAR_style", "Ridge_ext", "MedianReg_ext", "GBM_ext", "RandomForest", "SVR", "DynamicFactor", "GVAR_country", "Ridge_base", "MedianReg_base", "GBM_base", "RandomForest_base", "SVR_base", "LastGrowth", "Ridge_win", "MedianReg_win", "GBM_win", "SVR_win"):
    f = BASE / f"{k}.npz"
    if f.exists(): d = np.load(f); tab[k] = (d["test_pred"], float(np.abs(d["val_pred"] - d["val_true"]).mean()))

scale = np.tile(np.abs(G.loc[:, 1997:2014].values).mean(axis=1), len(YEARS)); scale = np.maximum(scale, 0.005)
rows, ens = [], {}
naive = np.abs(truth).mean()


def row(name, cfg, val, preds):
    P_ = np.stack(preds) if isinstance(preds, list) else preds[None]; e = np.abs(P_ - truth); m = e.mean(0)
    ens[name] = P_.mean(0)
    rows.append(dict(model=name, config=cfg, seeds=len(P_), val_MAE=val, test_MAE=e.mean(1).mean(), sd_seeds=e.mean(1).std() if len(P_) > 1 else np.nan,
                     RMSE=np.sqrt(((P_ - truth) ** 2).mean(1)).mean(), MASE=(m / scale).mean(), rel_to_nochange=m.mean() / naive, exCOVID=m[~cov].mean(), COVID=m[cov].mean(),
                     ensemble_MAE=np.abs(P_.mean(0) - truth).mean()))


for k, (cfg, val, preds, _) in models.items(): row(k, cfg, val, preds)
for k, (p, val) in tab.items(): row(k, "", val, p)
_rfw = [np.load(BASE / f"RandomForest_win_s{s_}.npz") for s_ in (42, 1, 2, 3, 4) if (BASE / f"RandomForest_win_s{s_}.npz").exists()]
if len(_rfw) == 5: row("RandomForest_win", "5 seeds", float(np.mean([np.abs(d["val_pred"] - d["val_true"]).mean() for d in _rfw])), [d["test_pred"] for d in _rfw])
T = pd.DataFrame(rows).sort_values("test_MAE"); T.to_csv(OUT / "final_accuracy.csv", index=False)
md = ["# Accuracy table (real GDP growth, test targets 2018-2024)", "",
      "MASE = MAE scaled by each country's in-sample (training-period) MAE of the no-change forecast; rel_to_nochange = MAE / MAE of the no-change forecast. Neural rows: mean over 5 seeds of the per-seed MAE (validation-selected config).", "",
      "```", T.round(4).to_string(index=False), "```", ""]

md += ["## Per-year test MAE (seed-mean predictions)", "", "```", pd.DataFrame({k: [np.abs(v - truth)[yr == t].mean() for t in YEARS] for k, v in ens.items()}, index=YEARS).T.round(4).loc[T.model].to_string(), "```", ""]

size = P[P.year == 2017].set_index("iso3").loc[order, "loggdp"].values; vol = G.loc[:, 1996:2017].std(axis=1).values
inc = (P[P.year == 2017].set_index("iso3").loc[order, "loggdp"] - P[P.year == 2017].set_index("iso3").loc[order, "logpop"]).values
tert = lambda x: np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]))
gr = {"size": (tert(size), ["small", "medium", "large"]), "volatility": (tert(vol), ["low", "medium", "high"]), "income(GDPpc proxy)": (tert(inc), ["low", "middle", "high"])}
keym = [k for k in ("STGNN", "STGNN_shuffled", "LSTM", "GRU", "GBM_win", "RandomForest_win", "SVR_win", "AR1_pooled", "ARIMA_rolling", "Persistence(level)") if k in ens]
for gname, (lab, nm) in gr.items():
    lt = np.tile(lab, len(YEARS)); md += [f"## Test MAE by country group: {gname} (terciles fixed with information up to 2017)", "", "```",
                                          pd.DataFrame({k: [np.abs(ens[k] - truth)[lt == j].mean() for j in range(3)] for k in keym}, index=nm).T.round(4).to_string(), "```", ""]

def dm(a, b):
    d = np.array([(np.abs(ens[a] - truth) - np.abs(ens[b] - truth))[yr == t].mean() for t in YEARS]); T_ = len(d)
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(T_)); hln = stat * np.sqrt((T_ - 1) / T_)
    return d.mean(), hln, 2 * stats.t.sf(abs(hln), T_ - 1), int((d < 0).sum())


for ref in ("STGNN",):
    if ref not in ens: continue
    md += [f"## Year-clustered Diebold-Mariano tests: {ref} vs each model (loss = |error|; negative mean diff = {ref} better; T = 7 test years, HLN-corrected, t(6))", "",
           "| Model | mean loss diff | DM stat | p | years better (of 7) |", "|---|---|---|---|---|"]
    for k in ens:
        if k == ref: continue
        m_, s_, p_, w_ = dm(ref, k); md.append(f"| {k} | {m_:+.5f} | {s_:+.2f} | {p_:.3f} | {w_} |")
    md.append("")
    md += ["Note: with only seven test years the DM test has little power; country-level paired errors (Wilcoxon) are reported in results_collected.md but treat countries within a year as dependent.", ""]
(OUT / "final_tables.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[:40]))
