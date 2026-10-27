"""Top-k sensitivity of the trade graph: the STGNN and its degree-preserving shuffled twin with k = 5, 25 and all destinations per exporter (the k = 12 runs
come from train_neural.py and 33_train_graph_controls.py). Same configuration (base features, h16, wd 1e-3), five seeds.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural, selection

if __name__ == "__main__":
    name, f, h, wd = selection.final_config("STGNN")
    jobs = []
    for k in ("k5", "k25", "kall"):
        for m in ("STGNN", "STGNN_shuffled"):
            nm = name if m == "STGNN" else name.replace("STGNN", "STGNN_shuffled", 1)
            jobs += [(selection.compose(nm, k), f, h, wd, s) for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
