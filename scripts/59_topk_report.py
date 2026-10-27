"""Top-k sensitivity report: full-network density, validation and test MAE of the STGNN and its degree-preserving shuffled twin for k = 5, 12, 25, all."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent)); import selection
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; RUNS = OUT / "runs"
N = 182; yr = np.repeat(range(2018, 2025), N)


def grp(name):
    nm, f, h, wd = selection.final_config("STGNN")
    base = nm if name.startswith("STGNN@") or name == "STGNN" else nm.replace("STGNN", "STGNN_shuffled", 1)
    suf = name.split("@", 1)[1] if "@" in name else None
    full = selection.compose(base, suf) if suf else base
    fs = [RUNS / f"{full}_{f}_h{h}_wd{wd}_s{s}.npz" for s in (42, 1, 2, 3, 4)]
    D = [np.load(f) for f in fs if f.exists()]
    return (np.stack([d["val_pred"] for d in D]), np.stack([d["test_pred"] for d in D]), D[0]["val_true"], D[0]["test_true"]) if len(D) == 5 else None


S = pd.read_csv(OUT / "graph_density_full_network.csv"); dens = S[S.year.between(2016, 2023)].mean(numeric_only=True)
rows = []
for k, suf, ke, cov in ((5, "@k5", "edges_k5", "cover_median_k5"), (12, "", "edges_k12", "cover_median_k12"), (25, "@k25", "edges_k25", "cover_median_k25"), ("all", "@kall", "edges_kall", None)):
    r, s_ = grp(f"STGNN{suf}"), grp(f"STGNN_shuffled{suf}")
    if r is None or s_ is None: continue
    er, es = np.abs(r[1] - r[3]).mean(0), np.abs(s_[1] - s_[3]).mean(0)
    d = np.array([(er - es)[yr == t].mean() for t in range(2018, 2025)]); hln = d.mean() / (d.std(ddof=1) / np.sqrt(7)) * np.sqrt(6 / 7)
    rows.append(dict(k=k, edges_per_year=int(dens[ke]), density=dens[ke] / (N * (N - 1)), export_coverage=dens[cov] if cov else 1.0, val_real=np.abs(r[0] - r[2]).mean(1).mean(), val_shuffled=np.abs(s_[0] - s_[2]).mean(1).mean(),
                     test_real=np.abs(r[1] - r[3]).mean(1).mean(), test_shuffled=np.abs(s_[1] - s_[3]).mean(1).mean(), diff=d.mean(), DM_p=2 * stats.t.sf(abs(hln), 6)))
T = pd.DataFrame(rows)
md = ["# Sensitivity to the number of destinations kept per exporter (STGNN, base features, h16, wd 1e-3, 5 seeds)", "",
      f"The complete BACI network among the 182 countries has directed density {dens['full_density']:.2f} (mean 2016-2023). The model graph keeps each exporter's k largest destinations.", "",
      "```", T.round(5).to_string(index=False), "```", "",
      "export_coverage = median share of an exporter's total exports captured by its top-k destinations; diff = mean (real - shuffled) test MAE; DM_p = year-clustered DM test (HLN, 7 years)."]
(OUT / "topk_sensitivity.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[4:10]))
