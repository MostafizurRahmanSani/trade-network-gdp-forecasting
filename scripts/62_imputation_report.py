"""Missing-data report and imputation sensitivity: counts of imputed cells per feature and per country group, and test errors on country-years whose input window
contains no imputed value versus those that do (same trained models, no retraining).
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, selection
import importlib.util
spec = importlib.util.spec_from_file_location("bp", ROOT / "scripts" / "02_build_panel.py"); bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
OUT = ROOT / "outputs"
wdi = bp.load_wdi(); P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
raw = {k: wdi[k].loc[order, 1994:2024] for k in ("infl", "emp")}
flag = {}
for k, df in raw.items():
    miss = df.isna(); interp = df.interpolate(axis=1, limit=3, limit_direction="both")
    flag[k] = {"missing": miss, "after_interp_missing": interp.isna()}
G = P.pivot(index="iso3", columns="year", values="g").loc[order]
size = P[P.year == 2017].set_index("iso3").loc[order, "loggdp"].values; inc = (P[P.year == 2017].set_index("iso3").loc[order, "loggdp"] - P[P.year == 2017].set_index("iso3").loc[order, "logpop"]).values
tert = lambda x: np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]))
groups = {"size": (tert(size), ["small", "medium", "large"]), "income(GDPpc proxy)": (tert(inc), ["low", "middle", "high"])}
cells = 182 * 29
md = ["# Missing values and imputation", "", f"Panel: {N} countries x 29 years (1996-2024) = {cells} country-years per feature. Real GDP and population are complete (countries with gaps were dropped).", "",
      "| Feature | raw missing cells | filled by <=3-year interpolation | filled by cross-country yearly median | share of country-years imputed |", "|---|---|---|---|---|"]
for k, nm in (("infl", "inflation (CPI, annual %)"), ("emp", "employment-to-population ratio (%)")):
    m = flag[k]["missing"].loc[:, 1996:2024].values.sum(); r = flag[k]["after_interp_missing"].loc[:, 1996:2024].values.sum()
    md.append(f"| {nm} | {m} | {m - r} | {r} | {m / cells:.1%} |")
for gname, (lab, names) in groups.items():
    md += ["", f"Imputed share of country-years by {gname} tercile:", "", "| group | inflation | employment ratio |", "|---|---|---|"]
    for j, nm in enumerate(names):
        md.append(f"| {nm} | " + " | ".join(f"{flag[k]['missing'].loc[:, 1996:2024].values[lab == j].mean():.1%}" for k in ('infl', 'emp')) + " |")

YEARS = list(range(2018, 2025)); wflag = []
for ty in YEARS:
    t = ty - 1; cols = list(range(t - 5, t + 1))
    wflag.append((flag["infl"]["missing"].loc[:, cols].values.any(axis=1)) | (flag["emp"]["missing"].loc[:, cols].values.any(axis=1)))
wflag = np.concatenate(wflag); yr = np.repeat(YEARS, N)
truth = np.load(OUT / "baselines" / "Persistence(level).npz")["test_true"]


def ens(pat):
    f = [x for x in sorted(glob.glob(str(OUT / "runs" / f"{pat}_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    return np.mean([np.load(x)["test_pred"] for x in f], 0)


models = {"STGNN": ens(selection.prefix("STGNN")), "STGNN shuffled graph": ens(selection.prefix("STGNN_shuffled")), "LSTM": ens(selection.prefix("LSTM")), "GRU": ens(selection.prefix("GRU"))}
for b in ("GBM_win", "RandomForest_win5", "SVR_win", "AR1_pooled", "Persistence(level)"):
    models[b] = np.load(OUT / "baselines" / f"{b}.npz")["test_pred"]
md += ["", f"Test samples whose input window contains at least one imputed inflation or employment value: {wflag.mean():.1%} ({int(wflag.sum())} of {len(wflag)}).", "",
       "Test MAE on samples with a complete window versus samples containing imputed values:", "", "| Model | complete windows | windows with imputed values | all |", "|---|---|---|---|"]
for k, p in models.items():
    e = np.abs(p - truth); md.append(f"| {k} | {e[~wflag].mean():.4f} | {e[wflag].mean():.4f} | {e.mean():.4f} |")
md += ["", "If the models were relying on imputed values, the ranking of models would change on the complete-window subset; the subset ranking is shown above."]
(OUT / "imputation_report.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[:40]))
