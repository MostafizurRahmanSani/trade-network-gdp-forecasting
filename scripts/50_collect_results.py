"""Collect the results of the study: test MAE overall / ex-COVID / COVID-only / per year, seed spread, paired tests and graph-control comparisons."""
import glob, json, re, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RUNS, BASE = ROOT / "outputs" / "runs", ROOT / "outputs" / "baselines"
YEARS = list(range(2018, 2025)); N = 182
yr = np.repeat(YEARS, N); cov = np.isin(yr, (2020, 2021))
sections = {"all": np.ones_like(cov), "ex-COVID": ~cov, "COVID(2020-21)": cov}


def grp(files):
    P = [np.load(f) for f in files]; return np.stack([p["test_pred"] for p in P]), P[0]["test_true"]


def load_nn():
    best = {}
    for fp in glob.glob(str(RUNS / "*.json")):
        n = Path(fp).stem
        if n.startswith("REFIT17_"): continue
        m = re.match(r"(.+?)_(base|ext|partner|leannode|lean)_h(\d+)_wd(\S+)_s(\d+)", n)
        best.setdefault(m.groups()[:4], []).append(fp)
    return {k: v for k, v in best.items() if len(v) >= 5}


def main():
    rows, preds = [], {}
    for k, fs in sorted(load_nn().items()):
        P, y = grp([f.replace(".json", ".npz") for f in fs])
        if y.size != 1274: continue
        vals = [json.load(open(f))["val_MAE"] for f in fs]
        nm = f"{k[0]} [{k[1]},h{k[2]},wd{k[3]}]"; preds[nm] = (P.mean(0), y)
        rows.append((nm, np.mean(vals), P, y))
    for f in sorted(BASE.glob("*.npz")):
        d = np.load(f)
        if "val_pred" not in d.files or d["test_pred"].size != 1274: continue
        preds[f.stem] = (d["test_pred"], d["test_true"]); rows.append((f.stem, np.abs(d["val_pred"] - d["val_true"]).mean(), d["test_pred"][None], d["test_true"]))
    out = ["# Results (real GDP growth; test targets 2018-2024)", "",
           "| Model | Val MAE | Test MAE all | ex-COVID | COVID 2020-21 | sd over seeds | seed-ensemble MAE |", "|---|---|---|---|---|---|---|"]
    for nm, v, P, y in sorted(rows, key=lambda r: np.mean(np.abs(r[2] - r[3]).mean(1))):
        e = np.abs(P - y)
        s = [e[:, m].mean(1) for m in sections.values()]
        out.append(f"| {nm} | {v:.4f} | {s[0].mean():.4f} | {s[1].mean():.4f} | {s[2].mean():.4f} | {s[0].std():.4f} | {np.abs(P.mean(0) - y).mean():.4f} |")
    out += ["", "## Per-year test MAE (seed-mean prediction)", "", "| Model | " + " | ".join(map(str, YEARS)) + " |", "|---|" + "---|" * len(YEARS)]
    for nm, (p, y) in preds.items():
        out.append(f"| {nm} | " + " | ".join(f"{np.abs(p - y)[yr == t].mean():.4f}" for t in YEARS) + " |")
    ref = "GBM_ext"; out += ["", f"## Paired comparison vs {ref} (|error| per country-year, all test years; Wilcoxon)", "", "| Model | mean diff (model - ref) | p |", "|---|---|---|"]
    er = np.abs(preds[ref][0] - preds[ref][1])
    for nm, (p, y) in preds.items():
        if nm == ref: continue
        d = np.abs(p - y) - er; out.append(f"| {nm} | {d.mean():+.4f} | {stats.wilcoxon(d).pvalue:.3g} |")
    (ROOT / "outputs" / "results_collected.md").write_text("\n".join(out)); print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
