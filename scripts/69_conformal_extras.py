"""Light computations on the saved conformal results (no fitting): (1) coverage and Winkler score without 2020-21 for all seven models, ranges across models used in the
text, and year-clustered DM tests on the yearly Winkler score (split CP, 90%); (2) median test-period growth (for the discussion of interval width); (3) self-loops and
duplicate edges of the randomised graph controls. Writes outputs/conformal_extras.md.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
OUT = ROOT / "outputs"; YRS = range(2018, 2025); EX = [2018, 2019, 2022, 2023, 2024]
R = pd.read_csv(OUT / "conformal_results.csv")
md = ["# Additional analyses of the conformal results (computed from saved outputs)", "", "## 1. Conformal results without 2020-2021 (mean of the five non-COVID test years)", "",
      "| alpha | model | method | coverage all | coverage excl. 2020-21 | Winkler all | Winkler excl. 2020-21 |", "|---|---|---|---|---|---|---|"]
for _, r in R[R.method.isin(["split-CP", "ACI", "rolling-CP", "native quantile"])].iterrows():
    md.append(f"| {r.alpha} | {r.model} | {r.method} | {r.coverage:.3f} | {np.mean([r[f'cov{t}'] for t in EX]):.3f} | {r.winkler:.3f} | {np.mean([r[f'wink{t}'] for t in EX]):.3f} |")
s = R[(R.alpha == 0.10) & (R.method == "split-CP")].set_index("model")
md += ["", "Year-clustered DM test on the yearly mean Winkler score (split CP, 90%, STGNN minus model; HLN, t(6)):", "", "| model | mean diff | p | years STGNN lower |", "|---|---|---|---|"]
w0 = np.array([s.loc["STGNN (5-seed mean)", f"wink{t}"] for t in YRS])
for m in s.index:
    if m.startswith("STGNN"): continue
    d = w0 - np.array([s.loc[m, f"wink{t}"] for t in YRS]); hln = d.mean() / (d.std(ddof=1) / np.sqrt(7)) * np.sqrt(6 / 7)
    md.append(f"| {m} | {d.mean():+.4f} | {2 * stats.t.sf(abs(hln), 6):.3f} | {int((d < 0).sum())} |")
for a in (0.10, 0.05):
    t = R[(R.alpha == a)]
    for meth in ("split-CP", "normalised-CP", "ACI", "rolling-CP"):
        u = t[t.method == meth]
        md.append(f"\nalpha {a} {meth}: coverage {u.coverage.min():.4f}-{u.coverage.max():.4f}; " + "; ".join(f"{y}: {u[f'cov{y}'].min():.4f}-{u[f'cov{y}'].max():.4f}" for y in YRS))
P, iso = V.load(); G = P.pivot(index="iso3", columns="year", values="g").loc[iso.iso3]
md += ["", "## 2. Test-period outcomes", "", f"Median real GDP growth over the 1,274 test country-years: {np.median(G.loc[:, 2018:2024].values):.4f} log points; median excluding 2020: {np.median(G.loc[:, [2018, 2019, 2021, 2022, 2023, 2024]].values):.4f}.", ""]
codes = iso.baci.tolist(); idx = {c: i for i, c in enumerate(codes)}; n = len(codes)
md += ["## 3. Self-loops and duplicate edges in the randomised graph controls (years 1996-2023, per seed; share of all edges)", "", "| control | seed | edges | self-loops | duplicate edges |", "|---|---|---|---|---|"]
for seed in (42, 1, 2, 3, 4):
    tot = {"shuffle": [0, 0, 0], "random": [0, 0, 0]}
    for y in range(1996, 2024):
        e = pd.read_csv(V.PROC / f"edges_{y}.csv.gz"); src = e.i.map(idx).values; dst = e.j.map(idx).values
        d1 = dst[np.random.RandomState(seed + y).permutation(len(dst))]
        rs = np.random.RandomState(seed + y); s2 = rs.randint(0, n, len(src)); d2 = rs.randint(0, n, len(dst))
        for k, (a, b) in (("shuffle", (src, d1)), ("random", (s2, d2))):
            pairs = pd.Series(list(zip(a, b))); tot[k][0] += len(a); tot[k][1] += int((a == b).sum()); tot[k][2] += int(pairs.duplicated().sum())
    for k, (m, sl, du) in tot.items():
        md.append(f"| {k} | {seed} | {m} | {sl} ({100 * sl / m:.2f}%) | {du} ({100 * du / m:.2f}%) |")
(OUT / "conformal_extras.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
