"""How much future information can the two-sided interpolation and the full-panel inflation winsorisation of 02_build_panel.py put into validation and test input
windows? Reads the raw WDI extract only; no model is fitted.

For every cell filled by df.interpolate(axis=1, limit=3, limit_direction="both") the observed years it depends on are recorded: interior linear interpolation uses the
previous and the next observation, a leading gap is back-filled from the first later observation, a trailing gap is forward-filled from the last earlier observation.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
iso = pd.read_csv(ROOT / "data" / "processed" / "countries.csv").iso3.tolist()
d = pd.read_csv(RAW / "wdi" / "WDI_extract.csv", na_values="..")
d = d[d["Series Code"].notna() & d["Country Code"].notna()]
d.columns = [c.split(" ")[0] if "[YR" in c else c for c in d.columns]
raw = {}
for code, nm in {"FP.CPI.TOTL.ZG": "infl", "SL.EMP.TOTL.SP.ZS": "emp"}.items():
    s = d[d["Series Code"] == code].set_index("Country Code")[[str(y) for y in range(1994, 2025)]]
    s.columns = range(1994, 2025); raw[nm] = s.astype(float).loc[iso]

AFF = []
SPL = {"validation (origins 2014-2016)": range(2014, 2017), "test (origins 2017-2023)": range(2017, 2024)}
md = ["# Look-ahead in the imputation of node features", "",
      "Raw WDI series, the 182 economies of the main sample. Window of origin t = input years t-5..t. 'future-dependent' = the filled value depends on an observation after the origin t.", ""]
for nm, df in raw.items():
    filled = df.interpolate(axis=1, limit=3, limit_direction="both")
    was_na = df.isna(); interp = was_na & filled.notna()
    rec = []
    for c in df.index:
        s = df.loc[c]; obs = s.dropna().index.values
        for y in interp.columns[interp.loc[c].values]:
            a = obs[obs < y].max() if (obs < y).any() else None; b = obs[obs > y].min() if (obs > y).any() else None
            lo_ = a if a is not None else df.columns.min() - 1; hi_ = b if b is not None else df.columns.max() + 1
            dep_future = b if (a is None or b is not None) else None
            if a is not None and b is None: dep_future = None
            rec.append(dict(c=c, y=int(y), a=a, b=b, kind="interior" if (a is not None and b is not None) else ("leading" if a is None else "trailing"),
                            gap=int(hi_ - lo_ - 1), val=filled.loc[c, y], fwd=(s.loc[a] if a is not None else np.nan), dep=dep_future))
    R = pd.DataFrame(rec)
    md += [f"## {nm}", "", f"Raw missing cells 1994-2024: {int(was_na.values.sum())}; filled by the interpolation: {len(R)} "
           f"(interior {int((R.kind == 'interior').sum())}, leading/back-filled {int((R.kind == 'leading').sum())}, trailing/forward-filled {int((R.kind == 'trailing').sum())}). "
           f"Filled cells lying in gaps longer than three years (pandas fills up to three cells from each end of such a gap): {int((R.gap > 3).sum())}.", ""]
    md += ["| split | window cells (country x year) | filled cells in windows | of which future-dependent | samples (country x origin) | samples with >=1 future-dependent cell | share of samples |", "|---|---|---|---|---|---|---|"]
    for lab, ts in SPL.items():
        n_cells = n_fill = n_fut = 0; samp = set()
        for t in ts:
            w = R[(R.y >= t - 5) & (R.y <= t)]
            n_cells += len(iso) * 6; n_fill += len(w)
            fut = w[w.dep.notna() & (w.dep > t)]; n_fut += len(fut); samp |= {(c, t) for c in fut.c}
        n_s = len(iso) * len(ts)
        AFF.extend(dict(feature=nm, split=lab.split()[0], iso3=c, origin=t, target=t + 1) for c, t in sorted(samp))
        md.append(f"| {lab} | {n_cells} | {n_fill} | {n_fut} | {n_s} | {len(samp)} | {100 * len(samp) / n_s:.1f}% |")
    fut_all = R[R.dep.notna()]
    if len(fut_all):
        inter = fut_all[fut_all.kind == "interior"]
        md += ["", f"Size of the issue: for interior filled cells, mean absolute difference between the interpolated value and a causal forward fill (last observed value): "
               f"{np.nanmean(np.abs(inter['val'] - inter['fwd'])):.2f} ({'percentage points of inflation' if nm == 'infl' else 'percentage points of the employment ratio'}); n = {len(inter)}.", ""]
    md += [f"Economies with future-dependent filled cells in any test window: {', '.join(sorted({c for t in SPL['test (origins 2017-2023)'] for c in R[(R.y >= t - 5) & (R.y <= t) & R.dep.notna() & (R.dep > t)].c}))}", ""]

infl = raw["infl"].interpolate(axis=1, limit=3, limit_direction="both"); infl = infl.apply(lambda c: c.fillna(c.median()), axis=0)
def transform(lo, hi):
    x = infl.clip(lo, hi); return np.sign(x) * np.log1p(x.abs() / 100)
full = np.nanpercentile(infl.values, [1, 99]); train = np.nanpercentile(infl.loc[:, 1994:2013].values, [1, 99])
A, B = transform(*full), transform(*train)
tr_cols = list(range(1996, 2014))
def clip_train(M):
    v = M.loc[:, tr_cols].values.ravel(); lo, hi = np.quantile(v, 0.01), np.quantile(v, 0.99); return M.clip(lo, hi)
A2, B2 = clip_train(A), clip_train(B)
diff = (np.abs(A2 - B2) > 1e-12)
md += ["## Inflation winsorisation", "", f"Thresholds (annual %): full panel 1994-2024 [{full[0]:.2f}, {full[1]:.2f}] (used); training input years 1994-2013 [{train[0]:.2f}, {train[1]:.2f}].",
       "After the training-period clipping applied in panel_data.build, the final inflation feature differs between the two versions in:", ""]
for lab, ts in SPL.items():
    yrs = sorted({y for t in ts for y in range(t - 5, t + 1)})
    md.append(f"- {lab}: {int(diff.loc[:, yrs].values.sum())} of {len(iso) * len(yrs)} country-year cells of the input years {yrs[0]}-{yrs[-1]}")
md.append(f"- all years 1996-2024: {int(diff.loc[:, 1996:2024].values.sum())} cells; largest absolute difference in the transformed feature: {float(np.abs(A2 - B2).values.max()):.4f}")
pd.DataFrame(AFF).to_csv(OUT / "imputation_lookahead_samples.csv", index=False)
(OUT / "imputation_lookahead.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
