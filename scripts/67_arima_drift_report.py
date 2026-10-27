"""MAE of the rolling ARIMA with and without drift on the IMF sample (as in 65_imf_comparison.py), per-year MAE, and the selected order.
Output: outputs/arima_drift_report.md (saved forecasts only).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, xlsx_reader as X
OUT = ROOT / "outputs"; B = OUT / "baselines"
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order); YEARS = list(range(2018, 2025)); yr = np.repeat(YEARS, N)
rows = X.read_sheet(str(ROOT / "data" / "external" / "weo" / "weohistorical.xlsx"), "xl/worksheets/sheet2.xml")
W = pd.DataFrame(rows[1:], columns=rows[0]).replace({".": np.nan, "": np.nan}); W["year"] = W["year"].astype(float)
imf = {}
for tag, col in (("fall", lambda Y: f"F{Y-1}ngdp_rpch"),):
    mat = np.full((len(YEARS), N), np.nan)
    for i, Y in enumerate(YEARS):
        c = col(Y)
        if c not in W.columns: continue
        sub = W[(W.year == Y) & W["ISOAlpha_3Code"].notna()].drop_duplicates("ISOAlpha_3Code").set_index("ISOAlpha_3Code")[c].astype(float)
        mat[i] = np.log1p(sub.reindex(order).values / 100)
    imf[tag] = mat.ravel()
ok = np.isin(yr, [2018, 2019, 2020, 2021, 2022, 2023]) & ~np.isnan(imf["fall"])
md = ["# Rolling ARIMA with drift", "", "| version | order | val MAE | test MAE | ex-COVID | COVID | " + " | ".join(str(y) for y in YEARS) + " | IMF sample all | IMF ex-COVID (2018-19, 2022-23) | IMF COVID |", "|---" * (11 + len(YEARS)) + "|"]
for lab, f in (("no drift (12_arima_rolling.py)", "ARIMA_rolling"), ("with drift (13_arima_drift.py)", "ARIMA_rolling_drift"), ("CP protocol, no drift", "ARIMA_cp"), ("CP protocol, with drift", "ARIMA_cp_drift")):
    d = np.load(B / f"{f}.npz"); e = np.abs(d["test_pred"] - d["test_true"]); c = np.isin(yr, (2020, 2021))
    o = tuple(d["order"]) if "order" in d.files else "see 12_arima_rolling.py log"
    m1 = ok; m2 = ok & np.isin(yr, (2018, 2019, 2022, 2023)); m3 = ok & np.isin(yr, (2020, 2021))
    md.append(f"| {lab} | {o} | {np.abs(d['val_pred'] - d['val_true']).mean():.5f} | {e.mean():.5f} | {e[~c].mean():.5f} | {e[c].mean():.5f} | " + " | ".join(f"{e[yr == y].mean():.4f}" for y in YEARS)
              + f" | {e[m1].mean():.4f} | {e[m2].mean():.4f} | {e[m3].mean():.4f} |")
md += ["", f"IMF October sample: {int(ok.sum())} country-years. Val MAE: main protocol targets 2015-17; CP protocol targets 2013-14."]
(OUT / "arima_drift_report.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
