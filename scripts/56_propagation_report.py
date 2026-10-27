"""Report of the shock-propagation experiment (scientific notation: the cross-country responses are numerically zero for the selected STGNN)."""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parents[1] / "outputs"
D = pd.read_csv(OUT / "propagation_raw.csv"); fmt = lambda x: f"{x:.2e}"
g = D.groupby(["mode", "shocked"])[["own", "receivers", "own_exporters", "non_neighbours"]].agg(["mean", "std"])
rows = []
for (mode, c), r in g.iterrows():
    rows.append(f"| {mode} | {c} | {r[('own','mean')]:+.4f} ± {r[('own','std')]:.4f} | {fmt(r[('receivers','mean')])} | {fmt(r[('own_exporters','mean')])} | {fmt(r[('non_neighbours','mean')])} |")
md = ["# Shock-propagation experiment (final STGNN configuration, 5 seeds per graph variant)", "",
      "A -5 percentage-point shock (log growth points) is injected into the 2019 growth input of one economy; the table gives the change in the predicted 2020 growth. "
      "'own' = the shocked economy itself (mean ± sd over seeds); the other columns are the mean ABSOLUTE change in the forecasts of (a) the economies that receive its exports (its 12 destinations in the real 2019 graph), "
      "(b) its own exporters (economies that list it as a destination), (c) all economies not linked to it. Only the growth input is shocked; derived partner-growth inputs are unchanged, so the response isolates message passing.", "",
      "| graph used for training/inference | shocked economy | own response | receivers (mean abs.) | exporters to it (mean abs.) | unlinked (mean abs.) |", "|---|---|---|---|---|---|"] + rows + ["",
      "Reading: the shocked economy's own forecast moves by about 0.5 times the shock (China -0.024 for a -0.05 shock), but forecasts of its trade partners change by less than 1e-8, i.e. numerically zero, for the real, the shuffled and the reversed graph alike. "
      "The trained STGNN therefore does not transmit information along trade edges; the temporal GRU per economy does all the work. Models without any cross-country input (LSTM, GRU, SVR, GBM, ...) have exactly zero cross-country response by construction."]
(OUT / "propagation.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[6:18]))
