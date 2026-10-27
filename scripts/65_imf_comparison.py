"""Compare the models with the IMF World Economic Outlook forecasts of real GDP growth (Historical WEO Forecasts Database, sheet ngdp_rpch).
For target year Y the October (Fall) vintage of year Y-1 is used (F{Y-1}); it is the closest analogue of the next-year forecast of the models.
IMF growth (%) is converted to log growth, log(1+g/100), and scored against the same WDI outcomes as every model. Target years 2018-2023 (the file ends with the
Fall 2022 vintage). The raw file must be downloaded first (see data/README.md).
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import xlsx_reader as X, panel_data as V, selection

rows = X.read_sheet(str(ROOT / "data" / "external" / "weo" / "weohistorical.xlsx"), "xl/worksheets/sheet2.xml")
W = pd.DataFrame(rows[1:], columns=rows[0]).replace({".": np.nan, "": np.nan})
W["year"] = W["year"].astype(float)
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
YEARS = list(range(2018, 2025))
imf = {}
for tag, col in (("IMF Fall(t) next-year", lambda Y: f"F{Y-1}ngdp_rpch"),):
    mat = np.full((len(YEARS), N), np.nan)
    for i, Y in enumerate(YEARS):
        c = col(Y)
        if c not in W.columns: continue
        sub = W[(W.year == Y) & W["ISOAlpha_3Code"].notna()].drop_duplicates("ISOAlpha_3Code").set_index("ISOAlpha_3Code")[c].astype(float)
        mat[i] = np.log1p(sub.reindex(order).values / 100)
    imf[tag] = mat.ravel()
y_true = np.load(ROOT / "outputs" / "baselines" / "Persistence(level).npz")["test_true"]
yr = np.repeat(YEARS, N)

R = ROOT / "outputs"
def ens(pat):
    f = [x for x in sorted(glob.glob(str(R / "runs" / f"{pat}_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    return np.mean([np.load(x)["test_pred"] for x in f], 0)
models = {"STGNN": ens(selection.prefix("STGNN")), "LSTM": ens(selection.prefix("LSTM")), "GRU": ens(selection.prefix("GRU"))}
for b in ("GBM_win", "RandomForest_win5", "SVR_win", "AR1_pooled", "ARIMA_rolling", "Persistence(level)", "RW_drift"):
    models[b] = np.load(R / "baselines" / f"{b}.npz")["test_pred"]
models.update({k: v for k, v in imf.items()})

out = ["# IMF WEO comparison (real GDP growth, log points; IMF growth % converted with log(1+g/100))", ""]
for label, years in (("target years 2018-2023 (IMF October forecast available)", [2018, 2019, 2020, 2021, 2022, 2023]), ("ex-COVID 2018-2019 and 2022-2023", [2018, 2019, 2022, 2023]), ("COVID 2020-2021", [2020, 2021])):
    m = np.isin(yr, years)
    ok = m.copy()
    for k in imf: ok &= ~np.isnan(imf[k])
    out += [f"## {label}; country-years with an IMF October forecast: {int(ok.sum())} country-years", "", "| Forecast | MAE | RMSE |", "|---|---|---|"]
    for k, v in sorted(models.items(), key=lambda kv: np.mean(np.abs(kv[1][ok] - y_true[ok]))):
        e = v[ok] - y_true[ok]; out.append(f"| {k} | {np.abs(e).mean():.4f} | {np.sqrt((e ** 2).mean()):.4f} |")
    out.append("")
out += ["Per-year MAE (common sample):", "", "| Forecast | " + " | ".join(str(y) for y in YEARS[:-1]) + " |", "|---|" + "---|" * (len(YEARS) - 1)]
ok = ~np.isnan(imf["IMF Fall(t) next-year"])
for k, v in models.items():
    out.append(f"| {k} | " + " | ".join(f"{np.abs(v[ok & (yr == y)] - y_true[ok & (yr == y)]).mean():.4f}" for y in YEARS[:-1]) + " |")
out += ["", "Notes: the Fall(t) vintage predates the full-year-t outturn, while the models use full-year-t data (through year t), so the information sets differ. "
        "Target 2024 is excluded because the historical-forecast file ends with the Fall 2022 vintage."]
(R / "imf_comparison.md").write_text("\n".join(out), encoding="utf-8"); print("\n".join(out[:40]))
