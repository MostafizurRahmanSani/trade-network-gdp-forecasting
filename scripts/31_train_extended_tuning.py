"""Extended tuning check around every family's selected configuration: learning rate (3e-4, 3e-3), hidden size (8 or 64) and weight decay (1e-2),
five seeds each. Decision rule fixed in advance: a new configuration is adopted only if its five-seed mean validation MAE is lower than the selected one by
more than 0.0003 (see 57_tuning_report.py and 32_adopt_extended_tuning.py).
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection, train_neural

FAMILIES = ["STGNN", "STGCN", "STGAT", "DCRNN", "GWN", "GNN", "GCN", "GAT", "GRU", "LSTM", "RNN", "BiLSTM", "Transformer", "TFT", "MTGNN", "MTGNN_trade"]

if __name__ == "__main__":
    jobs = []
    for m in FAMILIES:
        sel = selection.selected(m)
        if sel is None: print("no selection for", m, flush=True); continue
        f, h, wd = sel
        extra = [(f"{m}#lr0.0003", f, h, wd), (f"{m}#lr0.003", f, h, wd), (m, f, 8 if h == 16 else 64, wd), (m, f, h, "1e-2")]
        jobs += [(n, ff, hh, ww, s) for n, ff, hh, ww in extra for s in train_neural.SEEDS]
    with Pool(3) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
