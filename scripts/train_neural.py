"""Train and evaluate the neural models.

Each run is identified by (model, feature set, hidden size, weight decay, seed); a run that already has a result file is skipped. Validation MAE is used for
all selection; test predictions are saved with every run and never used for a choice.

Usage:  python scripts/train_neural.py grid --models GRU LSTM STGNN ...   (hidden size {16, 32} x weight decay {0, 1e-3}, five seeds each)
"""
import argparse, itertools, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
RUNS = ROOT / "outputs" / "runs"
EDGE_DIM, EPOCHS = 13, 300
SEEDS = (42, 1, 2, 3, 4)
TEMPORAL = ["GRU", "LSTM", "RNN", "BiLSTM", "Transformer", "TFT"]
SPATIAL = ["GNN", "GCN", "GAT", "BonnigGAT", "BonnigGAT_edges"]
LOSSES = ("l1", "huber")
split_loss = lambda n: (n[:-3], n[-2:]) if n.endswith("_l1") else (n[:-6], "huber") if n.endswith("_huber") else (n, "mse")


def parse_name(name):
    """-> (core model, loss, degree-preserving shuffle, graph mode)"""
    from panel_data import GMODES
    n, loss = split_loss(name)
    shuf = n.endswith("_shuffled"); n = n[:-9] if shuf else n
    gmode = "real"
    for m in GMODES:
        if n.endswith("_" + m): gmode, n = m, n[:-len(m) - 1]
    return n, loss, shuf, gmode


def build_model(name, F, h):
    base, _, _, _ = parse_name(name)
    from gdp_stgnn.models import GAT, GCN, GNN, GRU, LSTM, RNN, STGAT, STGCN, DCRNN, BiLSTM, GraphWaveNet
    if base.startswith("STGNNX"):
        from stgnn_variants import STGNNX, parse
        return STGNNX(F, h, EDGE_DIM, parse(base))
    k = dict(in_dim=F, hidden_dim=h, edge_dim=EDGE_DIM)
    if base == "Transformer":
        from stgnn_variants import TransformerBaseline
        return TransformerBaseline(hidden_dim=h, horizon=1, input_dim=F)
    if base in ("BonnigGAT", "BonnigGAT_edges"):
        from stgnn_variants import BonnigGAT
        return BonnigGAT(F, h, EDGE_DIM, use_edges=base.endswith("_edges"))
    if base == "TFT":
        from stgnn_variants import TFTBaseline
        return TFTBaseline(hidden_dim=h, horizon=1, input_dim=F)
    if base in ("MTGNN", "MTGNN_trade"):
        from stgnn_variants import MTGNN
        return MTGNN(F, h, EDGE_DIM, n_nodes=182, use_trade=base == "MTGNN_trade")
    if base in TEMPORAL: return {"GRU": GRU, "LSTM": LSTM, "RNN": RNN, "BiLSTM": BiLSTM}[base](hidden_dim=h, horizon=1, input_dim=F)
    if base in ("STGNN", "STGNN_uniform"):
        from stgnn_variants import STGNNX, parse
        return STGNNX(F, h, EDGE_DIM, parse("STGNNX-l1-evec-d2" if base == "STGNN" else "STGNNX-l1-euniform-d2"))
    return {"STGCN": lambda: STGCN(**k), "STGAT": lambda: STGAT(**k), "DCRNN": lambda: DCRNN(**k), "GWN": lambda: GraphWaveNet(**k, n_nodes=182),
            "GNN": lambda: GNN(**k), "GCN": lambda: GCN(**k), "GAT": lambda: GAT(**k)}[base]()


def predict_st(model, data):
    import torch
    from gdp_stgnn.data.graph_data import unpack_graphs
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for graphs, y in data:
            xs, ei, ea = unpack_graphs(graphs); ps.append(model(xs, ei, ea).numpy().ravel()); ts.append(y.numpy().ravel())
    return np.concatenate(ps), np.concatenate(ts)


def predict_gnn(model, data):
    import torch
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for g, y in data:
            ps.append(model(g.x, g.edge_index, g.edge_attr).numpy().ravel()); ts.append(y.numpy().ravel())
    return np.concatenate(ps), np.concatenate(ts)


def run_one(args):
    name, feats, h, wd, seed = args
    tag = f"{name}_{feats}_h{h}_wd{wd}_s{seed}"; f = RUNS / f"{tag}.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    if f.exists(): return tag
    import torch; torch.set_num_threads(1)
    from gdp_stgnn.config import spatial_only_hyperparams, stmodels_hyperparams, temporalonly_hyperparams
    from gdp_stgnn.data.sequence_data import build_gru_dataset_from_stgnn
    from gdp_stgnn.training import train_gnn, train_lstm, train_stmodel
    from gdp_stgnn.utils import set_seed
    import panel_data
    full_name = name; fold, norm, topk, ext_data, lr, horizon = None, "global", 12, False, None, 1
    if "#lr" in name:
        name, lr = name.split("#lr", 1); lr = float(lr)
    if "@" in name:
        name, ext = name.split("@", 1)
        if ext == "pynorm": norm = "peryear"
        elif ext in ("k5", "k25", "kall"): topk = {"k5": 5, "k25": 25, "kall": 0}[ext]
        elif ext == "ext": ext_data = True
        elif ext == "h2": horizon = 2
        else: fold = ext
    lrkw = {"learning_rate": lr} if lr else {}
    base, loss, shuf, gmode = parse_name(name)
    d = panel_data.build(feats, shuffle_edges=shuf, seed=seed, graph_mode=gmode, fold=fold, norm=norm, topk=topk, ext=ext_data, horizon=horizon)
    global EDGE_DIM
    EDGE_DIM = d["train"][0][0][0].edge_attr.size(1)
    F = d["train"][0][0][0].x.size(1); set_seed(seed); t0 = time.time()
    model = build_model(name, F, h)
    import torch.nn.functional as TF
    orig = TF.mse_loss
    if loss == "l1": TF.mse_loss = lambda p, y: TF.l1_loss(p, y)
    elif loss == "huber": TF.mse_loss = lambda p, y: TF.huber_loss(p, y, delta=0.03)
    if base in TEMPORAL:
        seq = [build_gru_dataset_from_stgnn(d[s], panel_data.WINDOW, 1) for s in ("train", "val", "test")]
        cal_seq = build_gru_dataset_from_stgnn(d["cal"], panel_data.WINDOW, 1) if d["cal"] else None
        hp = dict(temporalonly_hyperparams, n_epochs=EPOCHS, print_interval=10 ** 6, weight_decay=float(wd), **lrkw)
        model, _ = train_lstm(model, name, hp, seq[0][0], seq[0][1], seq[1][0], seq[1][1])
        model.eval()
        with torch.no_grad():
            vp, vt = model(torch.FloatTensor(seq[1][0])).numpy().ravel(), seq[1][1].ravel()
            tp, tt = model(torch.FloatTensor(seq[2][0])).numpy().ravel(), seq[2][1].ravel()
            cp_, ct_ = (model(torch.FloatTensor(cal_seq[0])).numpy().ravel(), cal_seq[1].ravel()) if cal_seq else (None, None)
    elif base in SPATIAL:
        sp = {s: [(gs[-1], y) for gs, y in d[s]] for s in ("train", "val", "test")}
        hp = dict(spatial_only_hyperparams, n_epochs=EPOCHS, print_interval=10 ** 6, weight_decay=float(wd), **lrkw)
        model, _ = train_gnn(model, name, hp, sp["train"], sp["val"])
        vp, vt = predict_gnn(model, sp["val"]); tp, tt = predict_gnn(model, sp["test"])
        cp_, ct_ = predict_gnn(model, [(gs[-1], y) for gs, y in d["cal"]]) if d["cal"] else (None, None)
    else:
        hp = dict(stmodels_hyperparams, n_epochs=EPOCHS, print_interval=10 ** 6, weight_decay=float(wd), **lrkw)
        model, _ = train_stmodel(model, name, hp, d["train"], d["val"])
        vp, vt = predict_st(model, d["val"]); tp, tt = predict_st(model, d["test"])
        cp_, ct_ = predict_st(model, d["cal"]) if d["cal"] else (None, None)
    TF.mse_loss = orig
    extra = {"cal_pred": cp_, "cal_true": ct_} if cp_ is not None else {}
    np.savez(RUNS / f"{tag}.npz", val_pred=vp, val_true=vt, test_pred=tp, test_true=tt, **extra)
    json.dump({"val_MAE": float(np.abs(vp - vt).mean()), "test_MAE": float(np.abs(tp - tt).mean()), "sec": time.time() - t0}, open(f, "w"))
    return tag


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["grid"]); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=3); a = ap.parse_args(); RUNS.mkdir(parents=True, exist_ok=True)
    jobs = [(m, "base", h, wd, s) for m in a.models for h in (16, 32) for wd in ("0", "1e-3") for s in SEEDS]
    with Pool(a.procs) as p:
        for tag in p.imap_unordered(run_one, jobs): print("done", tag, flush=True)
