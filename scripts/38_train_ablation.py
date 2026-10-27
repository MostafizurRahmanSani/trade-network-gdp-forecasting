"""One-factor ablation around the baseline configuration of the STGNN (base node features, h16, wd 1e-3, all 13 edge features), five seeds per variant.
Every variant is the same architecture with a single change; the variant with zero message-passing layers is trained with the same temporal GRU and the same procedure.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural, selection

SPECS = ["l0", "l1", "l3", "tlast", "tmean", "g2", "g3", "n0", "euniform", "elinear", "emlp4", "emlp16", "emlp32", "evec", "evecmlp", "eattn", "econcat", "cadd", "cconcat", "asum", "amax", "k1", "d2"]
_, FEAT, H0, WD = selection.final_config("STGNN")
VARS = [("STGNNX", H0)] + [(f"STGNNX-{s}", H0) for s in SPECS] + [("STGNNX", h) for h in (16, 32, 64) if h != H0]
if __name__ == "__main__":
    jobs = [(n, FEAT, h, WD, s) for n, h in VARS for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
