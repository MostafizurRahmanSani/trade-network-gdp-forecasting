"""Apply the decisions of the extended tuning check: trains five seeds of every adopted configuration (none is adopted in the reported results).
Reads outputs/tuning_decisions.json, which is written by 57_tuning_report.py.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_neural

if __name__ == "__main__":
    dec = json.load(open(Path(__file__).resolve().parents[1] / "outputs" / "tuning_decisions.json"))
    jobs = [(d["name"], d["feats"], int(d["h"]), d["wd"], s) for d in dec.values() for s in (42, 1, 2, 3, 4)]
    print(len(jobs), "jobs", flush=True)
    with Pool(4) as p:
        for t in p.imap_unordered(train_neural.run_one, jobs): print("done", t, flush=True)
