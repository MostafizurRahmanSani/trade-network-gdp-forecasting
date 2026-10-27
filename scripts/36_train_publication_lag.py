"""Publication-lag scenario: forecasts made without the last year's GDP, i.e. growth of year t+2 from a window ending in t (horizon 2).
Selected configurations, five seeds; the tabular and ARIMA counterparts are in 61_publication_lag_report.py.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural, selection

if __name__ == "__main__":
    jobs = []
    for m in ("STGNN", "STGNN_shuffled", "LSTM", "GRU", "MTGNN_trade"):
        base = "STGNN" if m == "STGNN_shuffled" else m
        name, f, h, wd = selection.final_config(base)
        if m == "STGNN_shuffled": name = name.replace("STGNN", "STGNN_shuffled", 1)
        jobs += [(selection.compose(name, "h2"), f, h, wd, s) for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
