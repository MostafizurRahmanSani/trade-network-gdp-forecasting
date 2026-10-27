"""Graph statistics per year and interpretation of the learned edge gates. Trains one STGNN (base features, h16, wd 1e-3, seed 42, 300 epochs) and reads its
edge-gate network on the 2023 graph. Single process.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import train_neural, panel_data
OUT = ROOT / "outputs"; PROC = ROOT / "data" / "processed"
iso = pd.read_csv(PROC / "countries.csv"); n = len(iso)
rows = []
for y in range(1996, 2024):
    e = pd.read_csv(PROC / f"edges_{y}.csv.gz"); outd = e.groupby("i").size().reindex(iso.baci).fillna(0).values; ind = e.groupby("j").size().reindex(iso.baci).fillna(0).values
    s = np.sort(ind)[::-1]; gini = (np.abs(ind[:, None] - ind[None]).sum() / (2 * n * ind.sum()))
    pairs = set(zip(e.i, e.j)); recip = np.mean([(j, i) in pairs for i, j in pairs])
    rows.append(dict(year=y, nodes=n, edges=len(e), density=len(e) / (n * (n - 1)), mean_out=outd.mean(), max_out=outd.max(), mean_in=ind.mean(), max_in=ind.max(),
                     isolated_in=int((ind == 0).sum()), in_degree_gini=gini, top5_share_of_in_edges=s[:5].sum() / ind.sum(), reciprocity=recip))
S = pd.DataFrame(rows); S.to_csv(OUT / "graph_statistics.csv", index=False)
md = ["# Graph statistics: each exporter keeps its top-12 destinations by total trade value", "", "```", S.round(3).to_string(index=False), "```", "",
      f"The graph is sparse (density {S.density.mean():.3f}, about {S.mean_out.mean():.1f} out-edges per country) with a highly skewed in-degree (max {int(S.max_in.max())}, Gini {S.in_degree_gini.mean():.2f}): "
      "a few large importers receive most edges. The fully connected alternative has density 1.0."]

import torch.nn.functional as F
from gdp_stgnn.config import stmodels_hyperparams
from gdp_stgnn.training import train_stmodel
from gdp_stgnn.utils import set_seed
import selection
_nm, FEAT, H0, WD0 = selection.final_config("STGNN")
d = panel_data.build(FEAT); train_neural.EDGE_DIM = 13; NF = d["train"][0][0][0].x.size(1)
P, _ = panel_data.load(); lg = P[P.year == 2023].set_index("iso3")["loggdp"]; code2iso = dict(zip(iso.baci, iso.iso3)); cols = panel_data.EDGE_COLS
e0 = pd.read_csv(PROC / "edges_2023.csv.gz")
for wd in sorted({float(WD0), 1e-3, 0.0}, reverse=True):
    set_seed(42); model = train_neural.build_model("STGNN", NF, H0)
    model, _ = train_stmodel(model, "STGNN-edgeweights", dict(stmodels_hyperparams, n_epochs=300, print_interval=10 ** 6, weight_decay=wd), d["train"], d["val"])
    g = d["test"][-1][0][-1]; ea = g.edge_attr
    with torch.no_grad(): w = model.layers[0].ew(ea).mean(1).numpy()
    nrm = lambda t: float(t.norm())
    wl = [(f"layer{i + 1}", nrm(L.upd.weight_ih), nrm(L.upd.weight_hh)) for i, L in enumerate(model.layers)]
    md += ["", f"Message-passing GRU weights (weight decay {wd}): " + "; ".join(f"{n}: ||W_message||={a:.4f}, ||W_state||={b:.4f}, ratio {a / max(b, 1e-12):.3f}" for n, a, b in wl) +
           ". W_message multiplies the aggregated neighbour message, W_state the node's own state; a ratio near 0 means the update ignores the neighbours."]
    Wdf = pd.DataFrame(ea.numpy(), columns=cols); Wdf["w"] = w; cor = Wdf.corr(method="spearman")["w"].drop("w")
    e = e0.copy(); e["w"] = w; e["imp_size"] = e.j.map(code2iso).map(lg); e["exp_size"] = e.i.map(code2iso).map(lg)
    md += ["", f"# Learned edge weights, weight decay {wd} (STGNN edge-weight network on the 2023 graph; weights in (0,1))", "",
           f"mean {w.mean():.5f}, sd {w.std():.6f}, min {w.min():.5f}, max {w.max():.5f}; share above their mean: {(w > w.mean()).mean():.2f}", "",
           "Spearman correlation of the learned weight with each (standardised) edge feature:", "", "```", cor.round(3).to_string(), "```", "",
           f"Spearman(weight, importer log GDP) = {e[['w','imp_size']].corr(method='spearman').iloc[0,1]:.3f}; Spearman(weight, exporter log GDP) = {e[['w','exp_size']].corr(method='spearman').iloc[0,1]:.3f}", "",
           "Top 5 and bottom 5 edges by learned weight (exporter -> importer):", "", "```",
           pd.concat([e.nlargest(5, "w"), e.nsmallest(5, "w")]).assign(exporter=lambda x: x.i.map(code2iso), importer=lambda x: x.j.map(code2iso))[["exporter", "importer", "v", "w"]].round(4).to_string(index=False), "```"]
(OUT / "graph_analysis.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[-22:]))
