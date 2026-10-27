"""Training curves of the STGNN (validation and test MAE every 10 epochs, mean +- sd over five seeds): 300 and 1000 epochs, plain training versus gradient clipping
with a cosine learning-rate schedule. Output: figures/fig_training_curves.png.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
for r, (plain, cc, lab) in enumerate((("e300", "e300_clip_cos", "300 epochs"), ("e1000", "e1000_clip_cos", "1000 epochs"))):
    for c, key, ttl in ((0, "val_curve_every10", "validation MAE"), (1, "test_curve_every10", "test MAE (reporting only; not used for any choice)")):
        ax = axes[r, c]
        for cfg, col, nm in ((plain, "tab:blue", "plain"), (cc, "tab:orange", "gradient clipping + cosine LR")):
            J = [json.load(open(f)) for f in glob.glob(str(OUT / "stability" / f"{cfg}_s*.json"))]
            if not J: continue
            A = np.array([j[key] for j in J]); ep = np.arange(A.shape[1]) * 10
            ax.plot(ep[2:], A.mean(0)[2:], color=col, label=nm); ax.fill_between(ep[2:], (A.mean(0) - A.std(0))[2:], (A.mean(0) + A.std(0))[2:], color=col, alpha=0.2)
        ax.set_title(f"{lab}: {ttl}", fontsize=8); ax.set_xlabel("epoch"); ax.set_ylabel("MAE (log growth points)")
        ax.set_ylim((0.0195, 0.0210) if c == 0 else (0.0349, 0.0358))
axes[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_training_curves.png", dpi=160); plt.close(fig)

T = pd.read_csv(OUT / "final_accuracy.csv").sort_values("test_MAE")
T = T[~T.model.isin(["RandomForest", "SVR", "GBM_ext", "Ridge_ext", "MedianReg_ext", "GBM_partner", "Ridge_partner", "MedianReg_partner"]) & ~T.model.str.endswith("_base")]
T = T.assign(model=T.model.str.replace("_win", "", regex=False).replace({"GBM": "GBM_ext", "Ridge": "Ridge_ext", "MedianReg": "MedianReg_ext"}))
NICE = {"RandomForest": "Random forest", "GBM_ext": "Gradient boosting", "GWN": "Graph WaveNet-style", "MTGNN": "MTGNN-style", "MTGNN_trade": "MTGNN-style (trade graph)",
        "TFT": "TFT-style", "Ridge_ext": "Ridge", "MedianReg_ext": "Median regression", "AR1+globalmean": "AR(1) + global mean", "AR1_pooled": "AR(1) pooled", "LastGrowth": "Last year's growth",
        "BonnigGAT": "Bonnig-style GAT", "BonnigGAT_edges": "Bonnig-style GAT (edge features)", "RW_drift": "Random walk with drift", "GVAR_style": "GVAR-type (pooled)",
        "PanelAR2_FE": "Panel AR(2), fixed effects", "GVAR_country": "GVAR-type (country)", "DynamicFactor": "Dynamic factor model", "CountryMeanGrowth": "Country mean growth",
        "ARIMA_rolling": "Rolling ARIMA", "Persistence(level)": "No-change"}
NICE.update({f"STGNN_{k}": f"STGNN, {v} graph" for k, v in (("shuffled", "shuffled"), ("uniform", "uniform-weight"), ("shufw", "shuffled-weight"), ("static", "static"), ("geo", "geographic"),
                                                            ("randedges", "random"), ("undirected", "undirected"), ("reversed", "reversed"))})
fig, ax = plt.subplots(figsize=(7, 0.22 * len(T) + 1.2))
col = ["tab:red" if m.startswith("STGNN") and "_" not in m or m == "STGNN" else "tab:grey" for m in T.model]
ax.barh([NICE.get(m, m) for m in T.model], T.test_MAE, xerr=T.sd_seeds.fillna(0), color=col); ax.invert_yaxis(); ax.set_xlim(0.033, 0.050); ax.set_xlabel("test MAE 2018-2024 (log growth points)")
fig.tight_layout(); fig.savefig(FIG / "fig_model_comparison.png", dpi=160); plt.close(fig)

R = pd.read_csv(OUT / "conformal_results.csv"); R = R[R.alpha == 0.10]
fig, ax = plt.subplots(figsize=(7, 3.8))
for (m, meth), ls in (((("STGNN (5-seed mean)", "split-CP"), "-"), (("STGNN (5-seed mean)", "ACI"), "--"), (("LSTM (5-seed mean)", "split-CP"), ":"), (("GBM", "split-CP"), "-."))):
    r = R[(R.model == m) & (R.method == meth)]
    if len(r): ax.plot(range(2018, 2025), [r[f"cov{t}"].iloc[0] for t in range(2018, 2025)], ls, marker="o", label=f"{m.split(' (')[0]} {meth}")
ax.axhline(0.9, color="k", lw=0.8); ax.set_ylim(0, 1.02); ax.set_xlabel("test year"); ax.set_ylabel("coverage (nominal 90%)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_conformal_coverage_by_year.png", dpi=160); plt.close(fig)

names = ["STGNN", "STGNN_shuffled", "STGNN_uniform", "STGNN_reversed", "STGNN_undirected", "STGNN_randedges", "STGNN_shufw", "STGNN_geo", "STGNN_static"]
sub = T.set_index("model").reindex(names).dropna(subset=["test_MAE"])
fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax.bar(range(len(sub)), sub.test_MAE, yerr=sub.sd_seeds, color="tab:blue"); ax.set_xticks(range(len(sub))); LAB = {"STGNN": "real trade graph", "STGNN_shuffled": "degree-preserving shuffle", "STGNN_uniform": "uniform weights", "STGNN_reversed": "reversed",
       "STGNN_undirected": "undirected", "STGNN_randedges": "random edges", "STGNN_shufw": "shuffled edge features", "STGNN_geo": "geographic", "STGNN_static": "static (2013)"}
ax.set_xticklabels([LAB[n] for n in sub.index], rotation=35, ha="right")
ax.set_ylim(0.034, 0.036); ax.set_ylabel("test MAE (log growth points)"); ax.set_title("STGNN with the trade graph and eight graph controls (5 seeds, mean +- sd)", fontsize=9); fig.tight_layout(); fig.savefig(FIG / "fig_graph_controls.png", dpi=160); plt.close(fig)
print("figures:", [p.name for p in FIG.glob("*.png")])
