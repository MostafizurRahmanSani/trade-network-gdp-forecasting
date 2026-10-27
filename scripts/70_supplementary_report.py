"""Report for the supplementary runs of 42_train_supplementary.py (graph controls of the STGNN without weight decay and of STGCN, the conformal-protocol selection, the
full-batch GRU/LSTM runs and the gate activations). No training here. Output: outputs/supplementary_report.md.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
OUT = ROOT / "outputs"; RUNS = OUT / "runs"; SEEDS = (42, 1, 2, 3, 4)
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
YEARS = list(range(2018, 2025)); yr = np.repeat(YEARS, N)
truth = np.load(OUT / "baselines" / "Persistence(level).npz")["test_true"]; truth_v = np.load(OUT / "baselines" / "Persistence(level).npz")["val_true"]


def load(prefix):
    D = [np.load(RUNS / f"{prefix}_s{s}.npz") for s in SEEDS]; J = [json.load(open(RUNS / f"{prefix}_s{s}.json")) for s in SEEDS]
    return np.stack([d["val_pred"] for d in D]), np.stack([d["test_pred"] for d in D]), J


def dm(pa, pb, y=truth, yrs=yr):
    d = np.array([(np.abs(pa.astype(float) - y) - np.abs(pb.astype(float) - y))[yrs == t].mean() for t in sorted(set(yrs))]); T = len(d)
    h = d.mean() / (d.std(ddof=1) / np.sqrt(T)) * np.sqrt((T - 1) / T); return d.mean(), 2 * stats.t.sf(abs(h), T - 1), int((d < 0).sum())


def summ(v, t):
    ev, et = np.abs(v - truth_v).mean(1), np.abs(t - truth).mean(1); return ev.mean(), et.mean(), et.std(ddof=1), np.abs(t.mean(0) - truth).mean()


md = ["# Supplementary runs (from saved runs)", ""]
md += ["## A. Graph controls on models that use the graph", "",
       "Per-seed means over 5 seeds; ens = MAE of the seed-averaged forecast; diff = real minus control on seed-averaged forecasts (negative: real graph better); DM: year-clustered, HLN, t(6); paired t over seeds on per-seed test MAE.", "",
       "| model | graph | val MAE | test MAE | seed sd | ens MAE | diff vs real | DM p | years real better | paired-seed p |", "|---|---|---|---|---|---|---|---|---|---|"]
prop_rows = []
for mdl, wd, orig in (("STGNN", "0", "STGNN_base_h16_wd0"), ("STGCN", "1e-3", "STGCN_base_h16_wd1e-3")):
    R = {}
    for mode, suf in (("real", ""), ("shuffled", "_shuffled"), ("randedges", "_randedges")):
        R[mode] = load(f"P1_{mdl}{suf}_base_h16_wd{wd}")
        for j, s in zip(R[mode][2], SEEDS):
            for p in j["propagation"]: prop_rows.append(dict(model=f"{mdl} wd{wd}", graph=mode, seed=s, **p))
    for mode in ("real", "shuffled", "randedges"):
        v, t, _ = R[mode]; a = summ(v, t)
        if mode == "real":
            md.append(f"| {mdl} (wd {wd}) | real | {a[0]:.5f} | {a[1]:.5f} | {a[2]:.5f} | {a[3]:.5f} | -- | -- | -- | -- |")
        else:
            m_, p_, w_ = dm(R["real"][1].mean(0), t.mean(0)); pt = stats.ttest_rel(np.abs(R["real"][1] - truth).mean(1), np.abs(t - truth).mean(1)).pvalue
            md.append(f"| {mdl} (wd {wd}) | {mode} | {a[0]:.5f} | {a[1]:.5f} | {a[2]:.5f} | {a[3]:.5f} | {m_:+.5f} | {p_:.3f} | {w_} | {pt:.3f} |")
    o = np.stack([np.load(RUNS / f"{orig}_s{s}.npz")["test_pred"] for s in SEEDS])
    md.append(f"| {mdl} (wd {wd}) | reproducibility: max abs difference between P1 real runs and the original runs {orig}: {np.abs(o - R['real'][1]).max():.2e} | | | | | | | | |")
md.append("")
D = pd.DataFrame(prop_rows); D.to_csv(OUT / "P1_propagation_raw.csv", index=False)
g = D.groupby(["model", "graph", "shocked"])[["own", "receivers", "own_exporters", "non_neighbours", "max_other"]].mean()
g["receivers/non-neighbours"] = g.receivers / g.non_neighbours
md += ["### Shock propagation (2019 growth input of the shocked economy set to -0.05; change in the 2020 forecast; mean over 5 seeds; mean absolute change for countries linked to it in the real 2019 graph)", "",
       "```", g.to_string(float_format=lambda x: f"{x:.2e}"), "```", ""]
G = P.pivot(index="iso3", columns="year", values="g")
md.append("Actual 2019 growth of the shocked economies (the shock replaces it with -0.05): " + ", ".join(f"{c} {G.loc[c, 2019]:+.4f}" for c in ("CHN", "USA", "DEU")))
old = pd.read_csv(OUT / "propagation_raw.csv"); og = old[old["mode"] == "real"].groupby("shocked")[["own", "receivers", "non_neighbours"]].mean()
md += ["", "Selected STGNN (wd 1e-3, propagation.md) for comparison:", "", "```", og.to_string(float_format=lambda x: f"{x:.2e}"), "```", ""]

md += ["## B. CP-protocol configurations selected on the early-stopping targets 2013-2014 only", "", "| model | hidden | wd | 5-seed MAE 2013-14 | calibration MAE 2015-17 | test MAE | selected (2013-14) | used before |", "|---|---|---|---|---|---|---|---|"]
before = {"STGNN": (16, "1e-3"), "GRU": (16, "1e-3"), "LSTM": (16, "1e-3")}
for m in ("STGNN", "GRU", "LSTM"):
    rows = []
    for h in (16, 32):
        for wd in ("0", "1e-3"):
            D_ = [np.load(RUNS / f"{m}@cp_base_h{h}_wd{wd}_s{s}.npz") for s in SEEDS]; J_ = [json.load(open(RUNS / f"{m}@cp_base_h{h}_wd{wd}_s{s}.json")) for s in SEEDS]
            rows.append((np.mean([j["val_MAE"] for j in J_]), h, wd, np.mean([np.abs(d["cal_pred"] - d["cal_true"]).mean() for d in D_]), np.mean([np.abs(d["test_pred"] - d["test_true"]).mean() for d in D_])))
    best = sorted(rows, key=lambda r: (round(r[0] / 1e-4), r[1], r[2] == "0"))[0]
    for r in rows:
        md.append(f"| {m} | {r[1]} | {r[2]} | {r[0]:.5f} | {r[3]:.5f} | {r[4]:.5f} | {'yes' if r is best else ''} | {'yes' if (r[1], r[2]) == before[m] else ''} |")
    if before[m][0] == 8:
        J_ = [json.load(open(RUNS / f"LSTM@cp_base_h16_wd1e-3_s{s}.json")) for s in SEEDS]; md.append(f"| LSTM | 8 | 1e-3 | {np.mean([j['val_MAE'] for j in J_]):.5f} | | | (not in the grid) | yes |")
md.append("")

md += ["## C. GRU and LSTM under the STGNN training protocol (full batch per training year, chronological, no clipping)", "",
       "| model | protocol | val MAE | test MAE | seed sd | ens MAE |", "|---|---|---|---|---|---|"]
E = {}
for lab, pre in (("GRU (h16)", "GRU_base_h16_wd1e-3"), ("GRU (h16)", "P3fb_GRU_base_h16_wd1e-3"), ("LSTM (h16)", "LSTM_base_h16_wd1e-3"), ("LSTM (h16)", "P3fb_LSTM_base_h16_wd1e-3"),
                 ("STGNN", "STGNN_base_h16_wd1e-3"), ("STGNN, 0 message-passing layers", "STGNNX-l0_base_h16_wd1e-3")):
    v, t, _ = load(pre); a = summ(v, t); E[pre] = t.mean(0)
    md.append(f"| {lab} | {'STGNN protocol' if pre.startswith('P3fb') or 'STGNN' in pre else 'mini-batch 32, clipping'} | {a[0]:.5f} | {a[1]:.5f} | {a[2]:.5f} | {a[3]:.5f} |")
md += ["", "| comparison (seed-averaged forecasts) | diff | DM p | years first better |", "|---|---|---|---|"]
for a_, b_ in (("P3fb_GRU_base_h16_wd1e-3", "GRU_base_h16_wd1e-3"), ("P3fb_LSTM_base_h16_wd1e-3", "LSTM_base_h16_wd1e-3"), ("STGNN_base_h16_wd1e-3", "P3fb_GRU_base_h16_wd1e-3"),
               ("STGNN_base_h16_wd1e-3", "P3fb_LSTM_base_h16_wd1e-3"), ("STGNNX-l0_base_h16_wd1e-3", "P3fb_GRU_base_h16_wd1e-3")):
    m_, p_, w_ = dm(E[a_], E[b_]); md.append(f"| {a_} vs {b_} | {m_:+.5f} | {p_:.3f} | {w_} |")
md.append("")

md += ["## D. Gate activations of the message-passing GRU cells (final STGNN, retrained with the train_neural procedure; test windows, all 6 years x 182 economies x 7 samples)", ""]
v, t, J = load("P9_STGNN_base_h16_wd1e-3"); o = np.stack([np.load(RUNS / f"STGNN_base_h16_wd1e-3_s{s}.npz")["test_pred"] for s in SEEDS])
md.append(f"Test MAE of the retrained models {np.abs(t - truth).mean(1).mean():.5f}; max abs difference to the original runs {np.abs(o - t).max():.2e}.")
rows = []
for s, j in zip(SEEDS, J):
    for l, gd in j["gates"].items(): rows.append(dict(seed=s, layer=l, **gd))
Gt = pd.DataFrame(rows); md += ["", "Mean over seeds (sd over seeds in the second table). update = z (PyTorch GRUCell: new state = (1 - z) * candidate + z * old state); reset = r; "
                                "message/state term = mean |W x| of the input (aggregated message) and hidden (own state) weight products, biases excluded.", "", "```",
                                Gt.drop(columns="seed").groupby("layer").mean().T.to_string(float_format=lambda x: f"{x:.4g}"), "```", "", "```",
                                Gt.drop(columns="seed").groupby("layer").std().T.to_string(float_format=lambda x: f"{x:.2g}"), "```", ""]
(OUT / "supplementary_report.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
