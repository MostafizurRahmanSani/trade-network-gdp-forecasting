"""Figures: (1) per-year test MAE of selected models and the IMF comparison, (2) trade-graph statistics (density of the complete network vs the top-12 graph;
in-degree distribution in 2019). Model errors come from the saved predictions; IMF per-year values from outputs/imf_comparison.md.
Outputs: figures/fig_per_year_mae.png, figures/fig_graph_statistics.png.
"""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; FIG = ROOT / "figures"; DATA = ROOT / "data" / "processed"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
YEARS = list(range(2018, 2025)); N = 182
truth = np.load(OUT / "baselines" / "Persistence(level).npz")["test_true"]


def ens(pat):
    f = [x for x in sorted(glob.glob(str(OUT / "runs" / f"{pat}_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    assert len(f) == 5, pat
    return np.mean([np.load(x)["test_pred"] for x in f], 0)


pred = {"STGNN": ens("STGNN_base_h16_wd1e-3"), "LSTM": ens("LSTM_base_h16_wd1e-3")}
for k, f in (("Random forest", "RandomForest_win5"), ("AR(1) pooled", "AR1_pooled"), ("Rolling ARIMA with drift", "ARIMA_rolling_drift"), ("No-change", "Persistence(level)")):
    pred[k] = np.load(OUT / "baselines" / f"{f}.npz")["test_pred"]

txt = (OUT / "imf_comparison.md").read_text(encoding="utf-8").split("Per-year MAE (common sample):")[1]
imf = {}
for line in txt.splitlines():
    m = re.match(r"\| (.+?) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
    if m: imf[m.group(1)] = [float(m.group(i)) for i in range(2, 8)]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
sty = {"STGNN": ("tab:red", "-", "o"), "Random forest": ("tab:green", "--", "s"), "LSTM": ("tab:blue", ":", "^"), "AR(1) pooled": ("tab:purple", "-.", "v"),
       "Rolling ARIMA with drift": ("tab:brown", "--", "d"), "No-change": ("k", "-", "x")}
for k, (c, ls, mk) in sty.items():
    e = np.abs(pred[k] - truth); main = k == "STGNN"
    axes[0].plot(YEARS, [e[i * N:(i + 1) * N].mean() for i in range(7)], ls, color=c, marker=mk, label=k, lw=2.8 if main else 1.1, ms=7 if main else 4,
                 zorder=3 if main else 4, alpha=1 if main else 0.85)
axes[0].set_title("(a) Test MAE by target year, all 182 economies"); axes[0].set_xlabel("target year"); axes[0].set_ylabel("MAE (log growth points)"); axes[0].legend(fontsize=7)
for k, lab, (c, ls, mk) in (("STGNN", "STGNN", sty["STGNN"]), ("RandomForest_win5", "Random forest", sty["Random forest"]), ("Persistence(level)", "No-change", sty["No-change"]),
                            ("IMF Fall(t) next-year", "IMF WEO", ("tab:blue", "-", "D"))):
    main = k == "STGNN"
    axes[1].plot(range(2018, 2024), imf[k], ls, color=c, marker=mk, label=lab, lw=2.8 if main else 1.1, ms=7 if main else 4, zorder=3 if main else 4, alpha=1 if main else 0.85)
axes[1].set_title("(b) IMF WEO October forecast (1,061 country-years)"); axes[1].set_xlabel("target year"); axes[1].set_xticks(range(2018, 2024)); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_per_year_mae.png", dpi=160); plt.close(fig)

yr = np.repeat(YEARS, N); covid = (yr == 2020) | (yr == 2021)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, sel, ttl in ((axes[0], ~covid, "(a) 2018-2019 and 2022-2024"), (axes[1], covid, "(b) 2020-2021")):
    ax.scatter(truth[sel], pred["STGNN"][sel], s=7, alpha=0.5, color="tab:red", label="STGNN")
    ax.scatter(truth[sel], np.zeros(sel.sum()), s=4, alpha=0.25, color="k", label="No-change (zero growth)")
    lim = (-0.3, 0.3); ax.plot(lim, lim, color="grey", lw=0.8); ax.set_xlim(lim); ax.set_ylim(-0.06, 0.12)
    ax.set_xlabel("actual growth (log points)"); ax.set_title(ttl)
axes[0].set_ylabel("predicted growth (log points)"); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_pred_vs_actual.png", dpi=160); plt.close(fig)

S = pd.read_csv(OUT / "graph_statistics.csv")
dens = {}
for line in (OUT / "graph_density_full_network.md").read_text(encoding="utf-8").splitlines():
    p = line.split()
    if len(p) == 9 and p[0].isdigit(): dens[int(p[0])] = float(p[2])
e = pd.read_csv(DATA / "edges_2019.csv.gz")
nodes = sorted(set(e.i) | set(e.j)); indeg = e.groupby("j").size().reindex(nodes, fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
axes[0].plot(list(dens), list(dens.values()), "o-", color="k", label="complete BACI network")
axes[0].plot(S.year, S.density, "s-", color="tab:blue", label="top-12 destinations per exporter (model graph)")
axes[0].set_ylim(0, 1); axes[0].set_xlabel("year"); axes[0].set_ylabel("directed density"); axes[0].set_title("(a) Graph density, 182 economies"); axes[0].legend(fontsize=7)
axes[1].hist(indeg.values, bins=np.arange(0, indeg.max() + 6, 5), color="tab:blue")
axes[1].set_yscale("log"); axes[1].set_xlabel("in-degree (exporters listing the economy in their top 12)"); axes[1].set_ylabel("number of economies")
axes[1].set_title(f"(b) In-degree, 2019 (max {indeg.max()}, {int((indeg == 0).sum())} economies with 0)")
fig.tight_layout(); fig.savefig(FIG / "fig_graph_statistics.png", dpi=160); plt.close(fig)
print("written", [p.name for p in FIG.glob("fig_*.png")])
