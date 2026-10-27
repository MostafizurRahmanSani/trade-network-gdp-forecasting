"""Sample-selection sensitivity: the selected configurations trained and evaluated on the extended country set (189 economies = main set + BTN, GRL, LBN, PLW,
SRB, TON, TUV), target years 2018-2023, five seeds.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural, selection

if __name__ == "__main__":
    jobs = []
    for m in ("STGNN", "STGNN_shuffled", "LSTM", "GRU", "SVR_placeholder"):
        if m == "SVR_placeholder": continue
        base = "STGNN" if m == "STGNN_shuffled" else m
        name, f, h, wd = selection.final_config(base)
        if m == "STGNN_shuffled": name = name.replace("STGNN", "STGNN_shuffled", 1)
        jobs += [(selection.compose(name, "ext"), f, h, wd, s) for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
