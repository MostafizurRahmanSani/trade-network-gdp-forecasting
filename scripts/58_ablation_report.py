"""One-factor ablation of the STGNN: validation MAE (five seeds) per variant, paired by seed against the baseline configuration; test MAE is reported only for the final model."""
import glob, json, re, sys
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
import train_neural, selection; train_neural.EDGE_DIM = 13
_nm, FEAT, H0, WD = selection.final_config("STGNN"); NF = {"base": 6, "ext": 11, "partner": 13}[FEAT]
cnt = lambda n, h: sum(p.numel() for p in train_neural.build_model(n, NF, h).parameters())


def runs(name, h):
    fs = [RUNS / f"{name}_{FEAT}_h{h}_wd{WD}_s{s}.json" for s in (42, 1, 2, 3, 4)]
    return [json.load(open(f)) for f in fs if f.exists()]


ref = runs("STGNNX", H0); rv = np.array([r["val_MAE"] for r in ref])
DESC = {"STGNNX": "final design (2 message-passing layers, GRU update, learned scalar edge weight, mean aggregation, L2-normalised, 1-layer temporal GRU)", "l0": "no message passing (0 layers): same temporal GRU on per-country features",
        "l1": "1 message-passing layer", "l3": "3 message-passing layers", "tlast": "no temporal integration: last input year only", "tmean": "temporal mean instead of GRU",
        "g2": "2-layer temporal GRU", "g3": "3-layer temporal GRU", "n0": "no L2 normalisation of node states", "euniform": "uniform edge weights (no edge features)", "elinear": "edge weight = sigmoid(linear(13 features))",
        "emlp4": "edge weight MLP 13->4->1", "emlp16": "edge weight MLP 13->16->1", "emlp32": "edge weight MLP 13->32->1", "evec": "per-channel gate from all 13 edge features",
        "evecmlp": "per-channel gate, 13->h->h MLP", "eattn": "attention over incoming edges (uses edge features)", "econcat": "edge features concatenated into the message", "cadd": "additive update instead of GRU update",
        "cconcat": "concatenation update", "asum": "sum aggregation", "amax": "max aggregation", "k1": "skip connection from the raw last-year features", "d2": "dropout 0.2"}
rows = []
for n, h in [("STGNNX", H0)] + [(f"STGNNX-{k}", H0) for k in list(DESC)[1:]] + [("STGNNX", hh) for hh in (16, 32, 64) if hh != H0]:
    r = runs(n, h)
    if len(r) < 5: continue
    v = np.array([x["val_MAE"] for x in r]); key = "STGNNX" if n == "STGNNX" else n.split("-")[1]
    d = v - rv; t = stats.ttest_rel(v, rv) if (n, h) != ("STGNNX", H0) else None
    rows.append((n, h, DESC.get(key, "hidden size %d" % h) if h == H0 or n != "STGNNX" else f"hidden size {h}", v.mean(), v.std(ddof=1), d.mean(), None if t is None else t.pvalue, cnt(n, h)))
rows.sort(key=lambda x: x[3])
md = ["# Ablation around the final model (final node feature set, all 13 edge features, final hidden size / weight decay, 5 seeds; validation MAE)", "",
      "Each row changes exactly one component of the final STGNN (reference row). Scores are VALIDATION MAE (mean and sd over 5 seeds); the difference to the reference is paired by seed (paired t-test over 5 seeds, uncorrected for the 25 comparisons). "
      "Test MAE is deliberately not shown for variants; the final model's test MAE is reported in final_tables.md.", "",
      "| Variant | hidden | what changes | val MAE | sd | diff vs final | p | parameters |", "|---|---|---|---|---|---|---|---|"]
for n, h, dsc, v, sdv, dd, p, c in rows:
    md.append(f"| {n.replace('STGNNX', 'final') if n != 'STGNNX' else 'final (reference)'} | {h} | {dsc} | {v:.5f} | {sdv:.5f} | {dd:+.5f} | {'-' if p is None else f'{p:.3g}'} | {c} |")
md += ["", f"Seed-to-seed sd of the reference validation MAE: {rv.std(ddof=1):.5f}. Differences smaller than about 2 x that value (0.0001-0.0002) are within noise."]
(OUT / "ablation_final.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[5:32]))
