"""Shock-propagation (perturbation) experiment. Trained STGNNs (real, degree-preserving shuffled and reversed graph; five seeds each) receive the 2019 test
window with a -5 percentage-point growth shock injected into the last-year growth feature of one large economy (China, USA, Germany; only the growth input is
shocked, derived partner-growth inputs are unchanged so that the measured response isolates message passing). The change in the predicted 2020 growth of all
other countries is measured and split by the relation to the shocked country in the real 2019 trade graph.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs"
SHOCK, COUNTRIES = -0.05, ("CHN", "USA", "DEU")


def job(args):
    mode, seed = args
    import torch, train_neural, panel_data
    from gdp_stgnn.config import stmodels_hyperparams
    from gdp_stgnn.data.graph_data import unpack_graphs
    from gdp_stgnn.training import train_stmodel
    from gdp_stgnn.utils import set_seed
    torch.set_num_threads(1)
    import selection
    nm, feat, hid, wd = selection.final_config("STGNN"); lr = float(nm.split("#lr")[1]) if "#lr" in nm else None
    d = panel_data.build(feat, shuffle_edges=(mode == "shuffled"), seed=seed, graph_mode="reversed" if mode == "reversed" else "real")
    real = panel_data.build(feat)
    train_neural.EDGE_DIM = 13; set_seed(seed)
    model = train_neural.build_model("STGNN", d["train"][0][0][0].x.size(1), hid)
    hp = dict(stmodels_hyperparams, n_epochs=300, print_interval=10 ** 6, weight_decay=float(wd), **({"learning_rate": lr} if lr else {}))
    model, _ = train_stmodel(model, "prop", hp, d["train"], d["val"]); model.eval()
    iso = d["iso"]; mu, sd = d["g_stats"]; k = 2
    xs, ei, ea = unpack_graphs(d["test"][k][0]); rei = real["test"][k][0][-1].edge_index
    res = []
    with torch.no_grad():
        base = model(xs, ei, ea).squeeze(-1)
        for c in COUNTRIES:
            ci = iso.index(c); x2 = xs.clone(); x2[-1, ci, 0] = (SHOCK - mu) / sd
            delta = (model(x2, ei, ea).squeeze(-1) - base).numpy()
            src, dst = rei[0].numpy(), rei[1].numpy()
            recv = np.unique(dst[src == ci]); srcs = np.unique(src[dst == ci]); other = np.setdiff1d(np.arange(len(iso)), np.r_[ci, recv, srcs])
            res.append(dict(mode=mode, seed=seed, shocked=c, own=float(delta[ci]), receivers=float(np.abs(delta[recv]).mean()), own_exporters=float(np.abs(delta[srcs]).mean()),
                            non_neighbours=float(np.abs(delta[other]).mean()), n_receivers=len(recv), n_exporters=len(srcs)))
    return res


if __name__ == "__main__":
    jobs = [(m, s) for m in ("real", "shuffled", "reversed") for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p: R = [r for rs in p.imap_unordered(job, jobs) for r in rs]
    D = pd.DataFrame(R); D.to_csv(OUT / "propagation_raw.csv", index=False)
    g = D.groupby(["mode", "shocked"]).mean(numeric_only=True).drop(columns="seed")
    g["receiver/non-neighbour"] = g.receivers / g.non_neighbours; g["exporter/non-neighbour"] = g.own_exporters / g.non_neighbours
    sd_ = D.groupby(["mode", "shocked"])[["own", "receivers", "own_exporters", "non_neighbours"]].std()
    md = ["# Shock-propagation experiment", "", f"A {SHOCK:+.2f} (log growth points) shock to the 2019 growth input of one country; change in predicted 2020 growth (log points) per 5-seed mean. "
          "'own' = shocked country's own forecast change; the other columns are mean absolute changes for countries that are receivers of its exports / its own exporters / not linked to it in the real 2019 graph.", "",
          "```", g[["own", "receivers", "own_exporters", "non_neighbours", "receiver/non-neighbour", "exporter/non-neighbour", "n_receivers", "n_exporters"]].round(5).to_string(), "```", "",
          "Seed standard deviations:", "", "```", sd_.round(5).to_string(), "```", "",
          "A model that exploits trade links should respond more at linked countries than at unlinked ones, and its response should exceed that of the shuffled-graph model (which has no real trade relations). "
          "A ratio close to 1 means the shock spreads (or fails to spread) uniformly. LSTM/GRU, SVR, GBM etc. have no cross-country input, so their cross-country response is exactly zero."]
    (OUT / "propagation.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[4:20]))
