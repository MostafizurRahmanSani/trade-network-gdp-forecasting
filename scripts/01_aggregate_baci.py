"""Aggregate the raw BACI HS92 (V202601) yearly files to exporter x importer x product-group trade values (thousand US$).

Product groups (HS chapters): 0 food/agriculture 01-24, 1 fuels 27, 2 ores/minerals 25-26, 3 chemicals 28-38, 4 plastics/wood/paper 39-49,
5 textiles/footwear 50-67, 6 stone/glass/precious 68-71, 7 base metals 72-83, 8 machinery/electronics 84-85, 9 transport/other 86-97.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "baci"
OUT = ROOT / "data" / "interim"


def group_of(chapter):
    g = np.full(chapter.shape, 9)
    for lo, hi, k in [(1, 24, 0), (27, 27, 1), (25, 26, 2), (28, 38, 3), (39, 49, 4), (50, 67, 5), (68, 71, 6), (72, 83, 7), (84, 85, 8)]:
        g[(chapter >= lo) & (chapter <= hi)] = k
    return g


def one(year):
    f = OUT / f"pair_group_{year}.csv.gz"
    if f.exists():
        return year
    b = pd.read_csv(RAW / f"BACI_HS92_Y{year}_V202601.csv", usecols=["i", "j", "k", "v"], dtype={"i": np.int32, "j": np.int32, "k": np.int64, "v": np.float64})
    b["g"] = group_of((b.k // 10000).values)
    a = b.groupby(["i", "j", "g"], as_index=False).v.sum()
    a.to_csv(f, index=False, compression="gzip")
    return year


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as p:
        for y in p.imap_unordered(one, range(1995, 2025)):
            print("done", y, flush=True)
