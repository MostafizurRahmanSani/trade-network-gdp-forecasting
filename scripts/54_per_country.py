"""Per-country test MAE for the STGNN, the shuffled-graph STGNN, LSTM, gradient boosting, AR(1) and no-change, and the countries where the STGNN gains or loses most versus AR(1)."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); import selection; OUT = ROOT / "outputs"
iso = pd.read_csv(ROOT / "data" / "processed" / "countries.csv"); N = len(iso)
def ens(pat):
    f = [x for x in sorted(glob.glob(str(OUT / "runs" / f"{pat}_s*.npz"))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    return np.mean([np.load(x)["test_pred"] for x in f], 0)
y = np.load(OUT / "baselines" / "Persistence(level).npz")["test_true"]
M = {"STGNN": ens(selection.prefix("STGNN")), "STGNN_shuffled": ens(selection.prefix("STGNN_shuffled")), "LSTM": ens(selection.prefix("LSTM"))}
for b in ("GBM_win", "AR1_pooled", "Persistence(level)"): M[b] = np.load(OUT / "baselines" / f"{b}.npz")["test_pred"]
D = pd.DataFrame({k: np.abs(v - y).reshape(-1, N).mean(0) for k, v in M.items()}, index=iso.iso3); D["STGNN-AR1"] = D["STGNN"] - D["AR1_pooled"]
D.round(5).to_csv(OUT / "per_country_mae.csv")
md = ["# Per-country test MAE (2018-24, seed-mean predictions)", "", f"STGNN beats AR(1) in {(D['STGNN-AR1'] < 0).sum()} of {N} countries and no-change in {(D['STGNN'] < D['Persistence(level)']).sum()}; "
      f"it beats LSTM in {(D['STGNN'] < D['LSTM']).sum()}. Real vs shuffled graph: STGNN better in {(D['STGNN'] < D['STGNN_shuffled']).sum()} countries, worse in {(D['STGNN'] > D['STGNN_shuffled']).sum()} (mean |difference| {np.abs(D['STGNN'] - D['STGNN_shuffled']).mean():.5f}).", "",
      "Largest gains over AR(1):", "", "```", D.nsmallest(8, "STGNN-AR1").round(4).to_string(), "```", "", "Largest losses versus AR(1):", "", "```", D.nlargest(8, "STGNN-AR1").round(4).to_string(), "```"]
(OUT / "per_country.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[:4]))
