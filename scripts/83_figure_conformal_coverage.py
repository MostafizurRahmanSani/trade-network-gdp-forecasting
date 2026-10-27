"""Coverage-by-year figure of the conformal intervals (split CP for the STGNN, LSTM and gradient boosting; ACI for the STGNN).
Output: figures/fig_conformal_coverage_by_year.png.
"""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
R = pd.read_csv(OUT / "conformal_results.csv"); R = R[R.alpha == 0.10]
fig, ax = plt.subplots(figsize=(7, 3.8))
for (m, meth), ls in (((("STGNN (5-seed mean)", "split-CP"), "-"), (("STGNN (5-seed mean)", "ACI"), "--"), (("LSTM (5-seed mean)", "split-CP"), ":"), (("GBM", "split-CP"), "-."))):
    r = R[(R.model == m) & (R.method == meth)]
    ax.plot(range(2018, 2025), [r[f"cov{t}"].iloc[0] for t in range(2018, 2025)], ls, marker="o", label=f"{m.split(' (')[0]} {meth}")
ax.axhline(0.9, color="k", lw=0.8); ax.set_ylim(0, 1.02); ax.set_xlabel("test year"); ax.set_ylabel("coverage (nominal 90%)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_conformal_coverage_by_year.png", dpi=160); plt.close(fig); print("written")
