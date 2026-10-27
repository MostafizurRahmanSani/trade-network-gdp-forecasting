"""Rolling-origin retraining of every neural family: origins 'gfc' (test targets 2009-10) and 'mid' (2013-16), selected configurations, five seeds.
The neural configurations are not re-tuned per origin.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural, selection

FAM = ["STGNN", "STGNN_shuffled", "STGCN", "STGAT", "DCRNN", "GWN", "GNN", "GCN", "GAT", "GRU", "LSTM", "RNN", "BiLSTM", "Transformer", "TFT", "MTGNN", "MTGNN_trade", "BonnigGAT", "BonnigGAT_edges"]
if __name__ == "__main__":
    jobs = []
    for m in FAM:
        base = "STGNN" if m == "STGNN_shuffled" else m
        cfg = selection.final_config(base)
        if cfg is None: print("no config", m); continue
        name, f, h, wd = cfg
        if m == "STGNN_shuffled": name = name.replace("STGNN", "STGNN_shuffled", 1)
        for fold in ("gfc", "mid"):
            jobs += [(selection.compose(name, fold), f, h, wd, s) for s in (42, 1, 2, 3, 4)]
    print(len(jobs), "jobs", flush=True)
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
