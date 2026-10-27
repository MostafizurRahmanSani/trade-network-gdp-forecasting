"""Runs for the conformal-protocol split ('@cp'), the per-year-normalisation check ('@pynorm') and the capacity checks, for the selected configurations
(five seeds).
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural, selection

if __name__ == "__main__":
    jobs = []
    for m in ("STGNN", "LSTM", "GRU"):
        name, f, h, wd = selection.final_config(m)
        for suf in ("cp", "pynorm"): jobs += [(selection.compose(name, suf), f, h, wd, s) for s in (42, 1, 2, 3, 4)]
    for m in ("LSTM", "GRU"):
        name, f, h, wd = selection.final_config(m)
        jobs += [(name, f, hh, wd, s) for hh in (48, 64) if hh != h for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
