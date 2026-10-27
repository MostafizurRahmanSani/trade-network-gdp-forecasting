"""Robustness checks from saved outputs only: (1) tabular regressors with three input sets (six features of year t plus four growth lags; 11 features without
partner growth; 13 features with trade-weighted partner growth), test and validation MAE and year-clustered DM test against the STGNN; (2) the refit on targets up to
2017 (43_refit_to_2017.py) against the fixed-origin models; (3) expanding-window and Mondrian split conformal prediction. Output: outputs/robustness_report.md.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
OUT = ROOT / "outputs"; RUNS = OUT / "runs"; B = OUT / "baselines"; SEEDS = (42, 1, 2, 3, 4)
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order); YEARS = list(range(2018, 2025)); yr = np.repeat(YEARS, N)
truth = np.load(B / "Persistence(level).npz")["test_true"].astype(float)
stg = np.mean([np.load(RUNS / f"STGNN_base_h16_wd1e-3_s{s}.npz")["test_pred"] for s in SEEDS], 0).astype(float)


def dm(a, b):
    d = np.array([(np.abs(a - truth) - np.abs(b - truth))[yr == t].mean() for t in YEARS]); h = d.mean() / (d.std(ddof=1) / np.sqrt(7)) * np.sqrt(6 / 7)
    return d.mean(), 2 * stats.t.sf(abs(h), 6), int((d < 0).sum())


md = ["# Robustness checks (round 3)", "", "## 1. Tabular regressors with three input sets (single fit, random_state 0 where random)", "",
      "| model | input set | file | val MAE | test MAE | DM p vs STGNN (STGNN better in years) |", "|---|---|---|---|---|---|"]
for lab, files in (("Ridge", ("Ridge_base", "Ridge_ext", "Ridge_partner")), ("Median regression", ("MedianReg_base", "MedianReg_ext", "MedianReg_partner")),
                   ("Gradient boosting", ("GBM_base", "GBM_ext", "GBM_partner")), ("Random forest", ("RandomForest_base", "RandomForest_ext", "RandomForest")),
                   ("SVR", ("SVR_base", "SVR_ext", "SVR"))):
    for setlab, f in zip(("6 features + 4 lags", "11 features + 4 lags", "13 features + 4 lags"), files):
        d = np.load(B / f"{f}.npz"); p = d["test_pred"].astype(float); m_, pv, w = dm(stg, p)
        md.append(f"| {lab} | {setlab} | {f} | {np.abs(d['val_pred'] - d['val_true']).mean():.5f} | {np.abs(p - truth).mean():.5f} | {pv:.3f} ({w}) |")
md.append("")

md += ["## 2. Refit on targets up to 2017 (fixed epoch budget = best validation epoch of the original run + 1)", "",
       "| model | fixed origin: test MAE (mean of seeds) | seed sd | seed-avg MAE | refit: test MAE (mean of seeds) | seed sd | seed-avg MAE | refit minus fixed (seed-avg) | DM p | years refit better | ex-COVID fixed / refit | COVID fixed / refit | best epochs |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
cov = np.isin(yr, (2020, 2021))
rows = {}
for lab, orig, refit in (("STGNN", "STGNN_base_h16_wd1e-3", "REFIT17_STGNN_base_h16_wd1e-3"), ("GRU", "GRU_base_h16_wd1e-3", "REFIT17_GRU_base_h16_wd1e-3"),
                         ("LSTM", "LSTM_base_h16_wd1e-3", "REFIT17_LSTM_base_h16_wd1e-3")):
    O = np.stack([np.load(RUNS / f"{orig}_s{s}.npz")["test_pred"] for s in SEEDS]).astype(float)
    F = [RUNS / f"{refit}_s{s}.npz" for s in SEEDS]
    if not all(f.exists() for f in F): md.append(f"| {lab} | missing refit runs |"); continue
    Rr = np.stack([np.load(f)["test_pred"] for f in F]).astype(float); ep = [json.load(open(RUNS / f"{refit}_s{s}.json"))["best_epoch_A"] for s in SEEDS]
    rows[lab] = (O.mean(0), Rr.mean(0))
    eo, er = np.abs(O - truth).mean(1), np.abs(Rr - truth).mean(1); m_, pv, w = dm(Rr.mean(0), O.mean(0))
    md.append(f"| {lab} | {eo.mean():.5f} | {eo.std(ddof=1):.5f} | {np.abs(O.mean(0) - truth).mean():.5f} | {er.mean():.5f} | {er.std(ddof=1):.5f} | {np.abs(Rr.mean(0) - truth).mean():.5f} | {m_:+.5f} | {pv:.3f} | {w} | "
              f"{np.abs(O.mean(0) - truth)[~cov].mean():.4f} / {np.abs(Rr.mean(0) - truth)[~cov].mean():.4f} | {np.abs(O.mean(0) - truth)[cov].mean():.4f} / {np.abs(Rr.mean(0) - truth)[cov].mean():.4f} | {ep} |")
O = np.stack([np.load(B / f"RandomForest_win_s{s}.npz")["test_pred"] for s in SEEDS]).astype(float); Rr = np.stack([np.load(B / f"REFIT17_RandomForest_s{s}.npz")["test_pred"] for s in SEEDS]).astype(float)
rows["Random forest"] = (O.mean(0), Rr.mean(0)); eo, er = np.abs(O - truth).mean(1), np.abs(Rr - truth).mean(1); m_, pv, w = dm(Rr.mean(0), O.mean(0))
md.append(f"| Random forest | {eo.mean():.5f} | {eo.std(ddof=1):.5f} | {np.abs(O.mean(0) - truth).mean():.5f} | {er.mean():.5f} | {er.std(ddof=1):.5f} | {np.abs(Rr.mean(0) - truth).mean():.5f} | {m_:+.5f} | {pv:.3f} | {w} | "
          f"{np.abs(O.mean(0) - truth)[~cov].mean():.4f} / {np.abs(Rr.mean(0) - truth)[~cov].mean():.4f} | {np.abs(O.mean(0) - truth)[cov].mean():.4f} / {np.abs(Rr.mean(0) - truth)[cov].mean():.4f} | config (3, 10, 0.6) |")
o = np.load(B / "AR1_pooled.npz")["test_pred"].astype(float); r = np.load(B / "REFIT17_AR1_pooled.npz")["test_pred"].astype(float); rows["AR(1) pooled"] = (o, r); m_, pv, w = dm(r, o)
md.append(f"| AR(1) pooled | {np.abs(o - truth).mean():.5f} | - | {np.abs(o - truth).mean():.5f} | {np.abs(r - truth).mean():.5f} | - | {np.abs(r - truth).mean():.5f} | {m_:+.5f} | {pv:.3f} | {w} | "
          f"{np.abs(o - truth)[~cov].mean():.4f} / {np.abs(r - truth)[~cov].mean():.4f} | {np.abs(o - truth)[cov].mean():.4f} / {np.abs(r - truth)[cov].mean():.4f} | coef {np.load(B / 'REFIT17_AR1_pooled.npz')['coef'].round(4).tolist()} |")
md += ["", "Per-year MAE, fixed origin -> refit (seed-averaged):", "", "| model | " + " | ".join(str(t) for t in YEARS) + " |", "|---|" + "---|" * 7]
for lab, (o, r) in rows.items():
    md.append(f"| {lab} | " + " | ".join(f"{np.abs(o - truth)[yr == t].mean():.4f} -> {np.abs(r - truth)[yr == t].mean():.4f}" for t in YEARS) + " |")
if "STGNN" in rows:
    md += ["", "Refit models compared with each other (seed-averaged, DM vs refit STGNN):", ""]
    for lab, (o, r) in rows.items():
        if lab == "STGNN": continue
        m_, pv, w = dm(rows["STGNN"][1], r); md.append(f"- refit STGNN vs refit {lab}: diff {m_:+.5f}, p {pv:.3f}, STGNN better in {w} years")
md.append("")

G = P.pivot(index="iso3", columns="year", values="g").loc[order]
TS = {"cal": range(2014, 2017), "test": range(2017, 2024)}; Y = {k: np.concatenate([G[t + 1].values for t in v]) for k, v in TS.items()}


def ens(prefix):
    D = [np.load(RUNS / f"{prefix}_s{s}.npz") for s in SEEDS]; return {"cal": np.mean([d["cal_pred"] for d in D], 0), "test": np.mean([d["test_pred"] for d in D], 0)}


pts = {"STGNN": ens("STGNN@cp_base_h32_wd1e-3"), "LSTM": ens("LSTM@cp_base_h16_wd1e-3"), "GRU": ens("GRU@cp_base_h16_wd1e-3")}
for k, f in (("Gradient boosting", "CPP2_GBM"), ("Random forest", "CPP2_RandomForest"), ("AR(1) pooled", "CPP2_AR1_pooled"), ("Rolling ARIMA with drift", "ARIMA_cp_drift")):
    if (B / f"{f}.npz").exists(): d = np.load(B / f"{f}.npz"); pts[k] = {"cal": d["cal_pred"], "test": d["test_pred"]}
vol = G.loc[:, 1996:2017].std(axis=1).values; vt = np.digitize(vol, np.quantile(vol, [1 / 3, 2 / 3])); gcal, gtest = np.tile(vt, 3), np.tile(vt, 7)
conf_q = lambda s, a: np.quantile(s, min(1.0, np.ceil((len(s) + 1) * (1 - a)) / len(s)), method="higher")
md += ["## 3. Expanding-window and Mondrian split CP for all seven models (90%)", "", "| model | split cov | expanding cov | expanding 2020 | ACI cov | Mondrian cov | Mondrian low/med/high | split low/med/high |", "|---|---|---|---|---|---|---|---|"]
R2 = pd.read_csv(OUT / "conformal_results.csv"); name = {"STGNN": "STGNN (5-seed mean)", "LSTM": "LSTM (5-seed mean)", "GRU": "GRU (5-seed mean)", "Gradient boosting": "GBM",
                                                             "Random forest": "RandomForest", "AR(1) pooled": "AR(1) pooled", "Rolling ARIMA with drift": "ARIMA rolling, drift"}
for k, v in pts.items():
    sc = np.abs(Y["cal"] - v["cal"]); yt = Y["test"]; pt = v["test"]
    q = conf_q(sc, .1); c_split = np.abs(yt - pt) <= q
    pool = [sc]; cov_e = np.zeros(len(yt), bool)
    for i in range(7):
        sl = slice(i * N, (i + 1) * N); qq = conf_q(np.concatenate(pool), .1); cov_e[sl] = np.abs(yt[sl] - pt[sl]) <= qq; pool.append(np.abs(yt[sl] - pt[sl]))
    cm = np.zeros(len(yt), bool)
    for g in range(3): cm[gtest == g] = np.abs(yt - pt)[gtest == g] <= conf_q(sc[gcal == g], .1)
    aci = R2[(R2.model == name[k]) & (R2.method == "ACI") & (R2.alpha == 0.1)].coverage.iloc[0]
    sp = R2[(R2.model == name[k]) & (R2.method == "split-CP") & (R2.alpha == 0.1)].coverage.iloc[0]
    md.append(f"| {k} | {c_split.mean():.3f} (file {sp:.3f}) | {cov_e.mean():.3f} | {cov_e[yr == 2020].mean():.3f} | {aci:.3f} | {cm.mean():.3f} | " + " / ".join(f"{cm[gtest == g].mean():.3f}" for g in range(3))
              + " | " + " / ".join(f"{c_split[gtest == g].mean():.3f}" for g in range(3)) + " |")
chk = OUT / "conformal_results.csv"
if chk.exists():
    A, Bc = pd.read_csv(OUT / "conformal_results.csv"), pd.read_csv(chk); num = A.select_dtypes("number").columns
    md += ["", f"Re-run check: max abs difference between conformal_results.csv and conformal_results.csv over all numeric cells: {np.nanmax(np.abs(A[num].values - Bc[num].values)):.2e}"]
(OUT / "robustness_report.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
