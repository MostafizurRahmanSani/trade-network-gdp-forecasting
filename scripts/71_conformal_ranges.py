"""Ranges across the seven conformal-protocol models (rolling ARIMA with drift) from outputs/conformal_results.csv (saved results, no fitting).
Output: outputs/conformal_ranges.md.
"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parents[1] / "outputs"
R = pd.read_csv(OUT / "conformal_results.csv"); R = R[R.model != "ARIMA rolling"]
YRS = range(2018, 2025); EX = [2018, 2019, 2022, 2023, 2024]
md = ["# CP ranges across the seven models of Table 12", ""]
for a in (0.10, 0.05):
    for m in ("split-CP", "normalised-CP", "ACI", "rolling-CP"):
        u = R[(R.alpha == a) & (R.method == m)]
        exc = u[[f"cov{t}" for t in EX]].mean(1)
        md.append(f"- alpha {a} {m}: models {len(u)}; coverage {u.coverage.min():.4f}-{u.coverage.max():.4f}; excl. 2020-21 {exc.min():.4f}-{exc.max():.4f}; Winkler {u.winkler.min():.4f}-{u.winkler.max():.4f}; "
                  + "; ".join(f"{t}: {u[f'cov{t}'].min():.4f}-{u[f'cov{t}'].max():.4f}" for t in YRS))
s = R[R.alpha == 0.10].pivot(index="model", columns="method", values="width")
md.append(f"- width ratio ACI/split: {(s['ACI'] / s['split-CP']).min():.3f}-{(s['ACI'] / s['split-CP']).max():.3f}; rolling/split: {(s['rolling-CP'] / s['split-CP']).min():.3f}-{(s['rolling-CP'] / s['split-CP']).max():.3f}")
w = R[R.alpha == 0.10].pivot(index="model", columns="method", values="winkler")
md.append("- Winkler ACI minus split: " + ", ".join(f"{k} {v:+.3f}" for k, v in (w["ACI"] - w["split-CP"]).dropna().items()))
md.append("- Winkler rolling minus split: " + ", ".join(f"{k} {v:+.3f}" for k, v in (w["rolling-CP"] - w["split-CP"]).dropna().items()))
c = R[R.alpha == 0.10].pivot(index="model", columns="method", values="cov2020")
md.append("- 2020 coverage ACI minus split: " + ", ".join(f"{k} {v:+.3f}" for k, v in (c["ACI"] - c["split-CP"]).dropna().items()))
sp = R[(R.alpha == 0.10) & (R.method == "split-CP")]
from scipy import stats
rho = stats.spearmanr(sp.mae, sp.winkler); md.append(f"- Spearman(MAE, split-CP Winkler) over these models: {rho.correlation:.2f}")
(OUT / "conformal_ranges.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
