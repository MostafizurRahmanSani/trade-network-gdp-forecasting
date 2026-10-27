"""Graph controls at the selected STGNN configuration, five seeds each: degree-preserving shuffle, uniform edge weights, shuffled edge features, random edges,
static graph (2013), geographic neighbours, reversed and undirected edges.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural, selection

if __name__ == "__main__":
    name, f, h, wd = selection.final_config("STGNN")
    lr = "#lr" + name.split("#lr")[1] if "#lr" in name else ""
    jobs = [(f"STGNN_{m}{lr}", f, h, wd, s) for m in ("shuffled", "uniform", "reversed", "undirected", "randedges", "shufw", "geo", "static") for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
