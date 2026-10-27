"""Validation-based selection helpers.

Every neural model uses the same six node features ('base': growth, log GDP, log population, population growth, inflation, employment ratio) and, for graph
models, all 13 edge features. Selection (validation MAE only, identical for every model family): the four configurations hidden size {16, 32} x weight decay
{0, 1e-3} are trained with five seeds (42, 1, 2, 3, 4); the configuration with the lowest five-seed mean validation MAE is the final one (means are rounded to
bins of 1e-4; ties are broken in favour of the smaller hidden size and then of weight decay > 0).
"""
import json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "outputs" / "runs"
GRID_H, GRID_WD, SEEDS = (16, 32), ("0", "1e-3"), (42, 1, 2, 3, 4)


def _val(name, cfg, seeds=SEEDS):
    """mean validation MAE of a configuration over the given seeds (None if a seed is missing)."""
    f, h, wd = cfg; fs = [RUNS / f"{name}_{f}_h{h}_wd{wd}_s{s}.json" for s in seeds]
    return float(np.mean([json.load(open(x))["val_MAE"] for x in fs])) if all(x.exists() for x in fs) else None


def selected(name):
    """final configuration (feature set, hidden size, weight decay): lowest five-seed mean validation MAE of the four grid configurations."""
    cand = []
    for h in GRID_H:
        for wd in GRID_WD:
            v = _val(name, ("base", h, wd))
            if v is not None: cand.append((round(v / 1e-4), h, wd == "0", v, ("base", h, wd)))
    if not cand: return None
    cand.sort(key=lambda r: r[:3]); return cand[0][4]


def final_config(m):
    """-> (run name, feature set, hidden, weight decay) of the configuration reported for family m: the protocol-selected one, unless the extended tuning check
    adopted a better one (outputs/tuning_decisions.json)."""
    dj = ROOT / "outputs" / "tuning_decisions.json"
    dec = json.load(open(dj)) if dj.exists() else {}
    if m in dec: d = dec[m]; return d["name"], d["feats"], int(d["h"]), d["wd"]
    sel = selected(m)
    return (m, *sel) if sel else None


def compose(name, suffix):
    """'STGNN#lr0.003' + 'gfc' -> 'STGNN@gfc#lr0.003' (the '@' part must precede the learning-rate override)."""
    if "#lr" in name:
        b, l = name.split("#lr", 1); return f"{b}@{suffix}#lr{l}"
    return f"{name}@{suffix}"


def prefix(m):
    """run-file prefix of the final configuration of family m (without the seed suffix); m may be 'STGNN_shuffled' (matched twin)."""
    base = "STGNN" if m == "STGNN_shuffled" else m
    rn, f, h, wd = final_config(base)
    if m == "STGNN_shuffled": rn = rn.replace("STGNN", "STGNN_shuffled", 1)
    return f"{rn}_{f}_h{h}_wd{wd}"
