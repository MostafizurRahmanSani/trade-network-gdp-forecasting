"""Shock-episode case studies from the saved predictions (no refitting): the real-minus-shuffled STGNN difference in MAE for the most exposed third of economies in
the 2015-16 commodity slowdown (validation years) and in 2020 (fuel exporters, and the economies with the weakest export-partner growth), with bootstrap intervals
over economies. Writes outputs/shock_cases.md.
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, selection
OUT = ROOT / "outputs"
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
Pw = lambda y: P[P.year == y].set_index("iso3").loc[order]


def ens(pat, key):
    f = [x for x in sorted(glob.glob(str(OUT / "runs" / f"{pat}_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    return np.mean([np.load(x)[key] for x in f], 0)


def episode(key, years_idx, exposure, most_exposed, label):
    r, s = ens(selection.prefix("STGNN"), key), ens(selection.prefix("STGNN_shuffled"), key)
    truth = np.load(OUT / "baselines" / "Persistence(level).npz")[key.replace("pred", "true")]
    tert = np.digitize(exposure, np.quantile(exposure, [1 / 3, 2 / 3])); j = 0 if most_exposed == "lowest" else 2
    sel = np.zeros(len(truth), bool)
    for yi in years_idx: sel[yi * N:(yi + 1) * N] = tert == j
    diff = (np.abs(r - truth) - np.abs(s - truth))[sel]
    rng = np.random.RandomState(0); bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return f"| {label} | {most_exposed} third | {int(sel.sum())} | {diff.mean():+.5f} | [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}] |"


md = ["# Shock episodes, corrected exposure groups (real-graph minus shuffled-graph STGNN MAE in the most exposed third; positive = real graph worse)", "",
      "| Episode | most exposed group | n | difference | bootstrap 95% interval |", "|---|---|---|---|---|",
      episode("val_pred", [0, 1], Pw(2014).fuel_share.values, "highest", "2015-16, fuel-export share (validation years)"),
      episode("test_pred", [2], Pw(2020).partner_ex_g.values, "lowest", "2020, realised partner growth (lowest = largest partner shock)"),
      episode("test_pred", [2], Pw(2020).partner_ex_g.values, "highest", "2020, realised partner growth, highest third (reference)"),
      episode("test_pred", [2], Pw(2019).fuel_share.values, "highest", "2020, fuel-export share")]
(OUT / "shock_cases.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
