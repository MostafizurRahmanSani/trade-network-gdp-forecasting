"""Supplementary training runs (one pool of 3 workers, one thread each; finished runs are skipped).
  P1  graph controls (real / degree-preserving shuffle / random edges) for the STGNN without weight decay (h16) and for STGCN (h16, wd 1e-3), five seeds, with the
      shock-propagation test of 40_shock_propagation.py on every trained model. Runs: runs/P1_<model>[_shuffled|_randedges]_base_h16_wd<wd>_s<seed>.*
  P2  conformal-protocol grid ({16,32} x {0,1e-3}) for STGNN, GRU, LSTM, five seeds, through train_neural.run_one (names STGNN@cp etc.)
  P3  GRU and LSTM trained with the STGNN procedure (one full-batch step per training year, chronological, no clipping). Runs: runs/P3fb_<model>_base_h<h>_wd1e-3_s<seed>.*
  P9  the selected STGNN retrained with the train_neural procedure, recording the GRU-cell gate activations on the test windows. Runs: runs/P9_STGNN_base_h16_wd1e-3_s<seed>.*
Usage: python scripts/42_train_supplementary.py P1 P2 P3 P9
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
RUNS = ROOT / "outputs" / "runs"
SEEDS = (42, 1, 2, 3, 4); SHOCK, COUNTRIES = -0.05, ("CHN", "USA", "DEU")


def _setup():
    import torch; torch.set_num_threads(1)
    import train_neural; train_neural.EDGE_DIM = 13
    return torch, train_neural


def _save(tag, vp, vt, tp, tt, t0, extra=None):
    np.savez(RUNS / f"{tag}.npz", val_pred=vp, val_true=vt, test_pred=tp, test_true=tt)
    js = {"val_MAE": float(np.abs(vp - vt).mean()), "test_MAE": float(np.abs(tp - tt).mean()), "sec": time.time() - t0}
    if extra: js.update(extra)
    json.dump(js, open(RUNS / f"{tag}.json", "w"))


def p1(model_name, wd, mode, seed):
    suffix = {"real": "", "shuffled": "_shuffled", "randedges": "_randedges"}[mode]
    tag = f"P1_{model_name}{suffix}_base_h16_wd{wd}_s{seed}"
    if (RUNS / f"{tag}.json").exists(): return tag
    torch, train_neural = _setup()
    import panel_data
    from gdp_stgnn.config import stmodels_hyperparams
    from gdp_stgnn.data.graph_data import unpack_graphs
    from gdp_stgnn.training import train_stmodel
    from gdp_stgnn.utils import set_seed
    d = panel_data.build("base", shuffle_edges=(mode == "shuffled"), seed=seed, graph_mode="randedges" if mode == "randedges" else "real")
    real = d if mode == "real" else panel_data.build("base")
    t0 = time.time(); set_seed(seed)
    model = train_neural.build_model(model_name, d["train"][0][0][0].x.size(1), 16)
    hp = dict(stmodels_hyperparams, n_epochs=300, print_interval=10 ** 6, weight_decay=float(wd))
    model, _ = train_stmodel(model, tag, hp, d["train"], d["val"])
    vp, vt = train_neural.predict_st(model, d["val"]); tp, tt = train_neural.predict_st(model, d["test"])
    iso = d["iso"]; mu, sd = d["g_stats"]; k = 2
    xs, ei, ea = unpack_graphs(d["test"][k][0]); rei = real["test"][k][0][-1].edge_index
    prop = []
    model.eval()
    with torch.no_grad():
        base = model(xs, ei, ea).squeeze(-1)
        for c in COUNTRIES:
            ci = iso.index(c); x2 = xs.clone(); x2[-1, ci, 0] = (SHOCK - mu) / sd
            delta = (model(x2, ei, ea).squeeze(-1) - base).numpy()
            src, dst = rei[0].numpy(), rei[1].numpy()
            recv = np.unique(dst[src == ci]); srcs = np.unique(src[dst == ci]); other = np.setdiff1d(np.arange(len(iso)), np.r_[ci, recv, srcs])
            prop.append(dict(shocked=c, own=float(delta[ci]), receivers=float(np.abs(delta[recv]).mean()), own_exporters=float(np.abs(delta[srcs]).mean()),
                             non_neighbours=float(np.abs(delta[other]).mean()), max_other=float(np.abs(np.delete(delta, ci)).max())))
    _save(tag, vp, vt, tp, tt, t0, {"propagation": prop})
    return tag


def p2(name, h, wd, seed):
    _setup()
    import train_neural
    return train_neural.run_one((name, "base", h, wd, seed))


class SeqAsST:
    pass


def p3(model_name, h, wd, seed):
    tag = f"P3fb_{model_name}_base_h{h}_wd{wd}_s{seed}"
    if (RUNS / f"{tag}.json").exists(): return tag
    torch, train_neural = _setup()
    import torch.nn as nn, panel_data
    from gdp_stgnn.config import stmodels_hyperparams
    from gdp_stgnn.training import train_stmodel
    from gdp_stgnn.utils import set_seed

    class Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, xs, edge_indices=None, edge_attrs=None):
            return self.inner(xs.permute(1, 0, 2))
    d = panel_data.build("base")
    t0 = time.time(); set_seed(seed)
    model = Wrap(train_neural.build_model(model_name, d["train"][0][0][0].x.size(1), h))
    hp = dict(stmodels_hyperparams, n_epochs=300, print_interval=10 ** 6, weight_decay=float(wd))
    model, _ = train_stmodel(model, tag, hp, d["train"], d["val"])
    vp, vt = train_neural.predict_st(model, d["val"]); tp, tt = train_neural.predict_st(model, d["test"])
    _save(tag, vp, vt, tp, tt, t0)
    return tag


def p9(seed):
    tag = f"P9_STGNN_base_h16_wd1e-3_s{seed}"
    if (RUNS / f"{tag}.json").exists(): return tag
    torch, train_neural = _setup()
    import torch.nn.functional as F, panel_data
    from gdp_stgnn.config import stmodels_hyperparams
    from gdp_stgnn.data.graph_data import unpack_graphs
    from gdp_stgnn.training import train_stmodel
    from gdp_stgnn.utils import set_seed
    d = panel_data.build("base")
    t0 = time.time(); set_seed(seed)
    model = train_neural.build_model("STGNN", d["train"][0][0][0].x.size(1), 16)
    hp = dict(stmodels_hyperparams, n_epochs=300, print_interval=10 ** 6, weight_decay=1e-3)
    model, _ = train_stmodel(model, tag, hp, d["train"], d["val"])
    vp, vt = train_neural.predict_st(model, d["val"]); tp, tt = train_neural.predict_st(model, d["test"])
    cap = {1: []}
    hooks = [model.layers[l - 1].upd.register_forward_hook(lambda m, inp, out, l=l: cap[l].append((inp[0].detach(), inp[1].detach()))) for l in (1,)]
    model.eval()
    with torch.no_grad():
        for graphs, _y in d["test"]:
            xs, ei, ea = unpack_graphs(graphs); model(xs, ei, ea)
    for hk in hooks: hk.remove()
    gates = {}
    for l in (1,):
        cell = model.layers[l - 1].upd
        a = torch.cat([c[0] for c in cap[l]]); hx = torch.cat([c[1] for c in cap[l]])
        gi = F.linear(a, cell.weight_ih, cell.bias_ih); gh = F.linear(hx, cell.weight_hh, cell.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
        r = torch.sigmoid(i_r + h_r); z = torch.sigmoid(i_z + h_z)
        wi = cell.weight_ih.chunk(3, 0); wh = cell.weight_hh.chunk(3, 0)
        a_term = torch.cat([F.linear(a, w) for w in wi], 1).abs().mean().item()
        h_term = torch.cat([F.linear(hx, w) for w in wh], 1).abs().mean().item()
        gates[f"layer{l}"] = dict(update_mean=z.mean().item(), update_sd=z.std().item(), update_min=z.min().item(), update_max=z.max().item(),
                                  reset_mean=r.mean().item(), reset_sd=r.std().item(), mean_abs_message_term=a_term, mean_abs_state_term=h_term,
                                  mean_abs_bias_ih=cell.bias_ih.abs().mean().item(), mean_abs_bias_hh=cell.bias_hh.abs().mean().item(),
                                  norm_W_ih=cell.weight_ih.norm().item(), norm_W_hh=cell.weight_hh.norm().item(), n=int(a.shape[0]))
    _save(tag, vp, vt, tp, tt, t0, {"gates": gates})
    return tag


def run(job):
    kind, args = job
    t = time.time()
    try:
        out = {"P1": p1, "P2": p2, "P3": p3, "P9": p9}[kind](*args)
        return f"{kind} {out} done in {time.time() - t:.0f}s"
    except Exception as e:
        import traceback
        return f"{kind} {args} FAILED: {e!r}\n{traceback.format_exc()}"


if __name__ == "__main__":
    jobs = []
    for s in SEEDS:
        for mode in ("real", "shuffled", "randedges"):
            jobs.append(("P1", ("STGNN", "0", mode, s))); jobs.append(("P1", ("STGCN", "1e-3", mode, s)))
    for name in ("STGNN@cp", "GRU@cp", "LSTM@cp"):
        for h in (16, 32):
            for wd in ("0", "1e-3"):
                for s in SEEDS: jobs.append(("P2", (name, h, wd, s)))
    for s in SEEDS:
        jobs.append(("P3", ("GRU", 16, "1e-3", s))); jobs.append(("P3", ("LSTM", 16, "1e-3", s))); jobs.append(("P9", (s,)))
    sel = sys.argv[1:]
    if sel: jobs = [j for j in jobs if j[0] in sel]
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(3, maxtasksperchild=4) as p:
        for msg in p.imap_unordered(run, jobs): print(time.strftime("%H:%M:%S"), msg, flush=True)
    print("ALL DONE", flush=True)
