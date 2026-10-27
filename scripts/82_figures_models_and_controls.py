"""Figures from saved outputs only:
  fig_model_comparison.png   dot plot of test MAE of all models, seed sd as error bars
  fig_graph_controls.png     dot plot of the STGNN with the real graph and the eight controls
  fig_pred_vs_actual.png     predicted vs actual growth of the STGNN, equal axis scales, negative outcomes marked
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
T = pd.read_csv(OUT / "point_accuracy.csv").set_index("key")
SHOW = [k for k in T.index if not k.startswith("STGNNX")]
T = T.loc[SHOW].sort_values("test_MAE")
fig, ax = plt.subplots(figsize=(6.6, 0.2 * len(T) + 1.0))
y = np.arange(len(T))
for i, (k, r) in enumerate(T.iterrows()):
    c = "tab:red" if k == "STGNN" else ("tab:orange" if k.startswith("STGNNX") else "tab:grey")
    ax.errorbar(r.test_MAE, i, xerr=0 if np.isnan(r.sd) else r.sd, fmt="o", color=c, ms=4, capsize=2)
ax.set_yticks(y); ax.set_yticklabels([m.replace("Growth persistence (last year's growth)", "Last year's growth").replace("Bonnig", "Bönnig") for m in T.model], fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("test MAE 2018-2024 (log growth points)")
fig.tight_layout(); fig.savefig(FIG / "fig_model_comparison.png", dpi=170); plt.close(fig)

truth = np.load(OUT / "baselines" / "Persistence(level).npz")["test_true"]
LAB = [("STGNN", "Real trade graph"), ("STGNN_shuffled", "Degree-preserving shuffle"), ("STGNN_uniform", "Uniform edge weights"), ("STGNN_shufw", "Shuffled edge features"),
       ("STGNN_randedges", "Random edges"), ("STGNN_static", "Static (2013) graph"), ("STGNN_geo", "Geographic neighbours"), ("STGNN_reversed", "Reversed"), ("STGNN_undirected", "Undirected")]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for i, (k, lab) in enumerate(LAB):
    e = np.array([np.abs(np.load(f)["test_pred"] - truth).mean() for f in sorted(glob.glob(str(OUT / "runs" / f"{k}_base_h16_wd1e-3_s*.npz"))) if f.rsplit("_s", 1)[1][:-4].isdigit()])
    ax.errorbar(e.mean(), i, xerr=e.std(), fmt="o", color="tab:red" if i == 0 else "tab:blue", capsize=3)
    ax.plot(e, [i] * len(e), "|", color="grey", ms=6)
ax.set_yticks(range(len(LAB))); ax.set_yticklabels([l for _, l in LAB]); ax.invert_yaxis()
ax.set_xlim(0.0345, 0.0360); ax.set_xlabel("test MAE 2018-2024 (log growth points)")
fig.tight_layout(); fig.savefig(FIG / "fig_graph_controls.png", dpi=170); plt.close(fig)

f = [x for x in sorted(glob.glob(str(OUT / "runs" / "STGNN_base_h16_wd1e-3_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
p = np.mean([np.load(x)["test_pred"] for x in f], 0); yr = np.repeat(range(2018, 2025), 182); covid = np.isin(yr, (2020, 2021))
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
lim = (-0.40, 0.40)
for ax, sel, ttl in ((axes[0], ~covid, "(a) 2018-2019 and 2022-2024"), (axes[1], covid, "(b) 2020-2021")):
    neg = sel & (truth < 0); pos = sel & (truth >= 0)
    ax.scatter(truth[pos], p[pos], s=7, alpha=0.5, color="tab:red", label="STGNN, outcome >= 0")
    ax.scatter(truth[neg], p[neg], s=9, alpha=0.7, color="tab:blue", marker="v", label="STGNN, outcome < 0")
    ax.plot(lim, lim, color="grey", lw=0.8); ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.set_xlabel("actual growth (log points)"); ax.set_title(ttl)
    out_ = int((sel & ((truth < lim[0]) | (truth > lim[1]))).sum())
    if out_: ax.text(0.02, 0.02, f"{out_} outcome(s) outside the axis range", transform=ax.transAxes, fontsize=7)
axes[0].set_ylabel("predicted growth (log points)"); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_pred_vs_actual.png", dpi=170); plt.close(fig)
print("written")
