"""Sample-selection sensitivity: adding the seven economies that miss only boundary years (BTN, GRL, LBN, PLW, SRB, TON, TUV). Models are retrained on the extended
panel (189 economies, selected configurations, five seeds) and evaluated on target years 2018-2023 (2024 is unavailable for six of the additions).
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, selection
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
P0, iso0 = V.load(); Pe, isoe = V.load(V.PROC_EXT); n0, ne = len(iso0), len(isoe)
added = np.array([c not in set(iso0.iso3) for c in isoe.iso3]); YEARS = list(range(2018, 2024))


def runs(name, ext):
    f, h, wd = name[1:] if False else (None, None, None)
    base = "STGNN" if name == "STGNN_shuffled" else name
    rn, f, h, wd = selection.final_config(base)
    if name == "STGNN_shuffled": rn = rn.replace("STGNN", "STGNN_shuffled", 1)
    pat = f"{selection.compose(rn, 'ext') if ext else rn}_{f}_h{h}_wd{wd}"
    fs = [x for x in sorted(glob.glob(str(RUNS / (pat + "_s*.npz")))) if x.rsplit("_s", 1)[1][:-4].isdigit()]
    return np.stack([np.load(x)["test_pred"] for x in fs]) if len(fs) >= 5 else None


def growth(P, iso):
    return P.pivot(index="iso3", columns="year", values="g").loc[iso.iso3]


G0, Ge = growth(P0, iso0), growth(Pe, isoe)
y0 = np.concatenate([Ge[t].values[~added] for t in YEARS]) if False else np.concatenate([G0[t].values for t in range(2018, 2024)])
ye = np.concatenate([Ge[t].values for t in YEARS]); yr_e = np.repeat(YEARS, ne); add_e = np.tile(added, len(YEARS))
rows = []
lo, hi = np.percentile(np.concatenate([Ge[t + 1].values for t in range(2001, 2014)]), [1, 99])
A = lambda ts: np.nan_to_num(np.column_stack([np.concatenate([Ge[t].values for t in ts]), np.ones(ne * len(ts))]))
b = np.linalg.lstsq(A(range(2001, 2014)), np.clip(np.concatenate([Ge[t + 1].values for t in range(2001, 2014)]), lo, hi), rcond=None)[0]
ar1_e = A(range(2017, 2023)) @ b
cands = {"No-change": (np.zeros(len(ye)), np.zeros(len(y0))), "AR(1) pooled": (ar1_e, np.load(OUT / "baselines" / "AR1_pooled.npz")["test_pred"][: n0 * 6])}
for m in ("STGNN", "STGNN_shuffled", "LSTM", "GRU"):
    pe, p0 = runs(m, True), runs(m, False)
    if pe is not None and p0 is not None: cands[m] = (pe.mean(0), p0.mean(0)[: n0 * 6])
for m, (pe, p0) in cands.items():
    rows.append(dict(model=m, main_sample_182=np.abs(p0 - y0).mean(), extended_sample_189=np.abs(pe - ye).mean(), only_182_original=np.abs(pe - ye)[~add_e].mean(), only_7_added=np.abs(pe - ye)[add_e].mean()))
T = pd.DataFrame(rows).set_index("model")
md = ["# Sample-selection sensitivity (targets 2018-2023)", "", "Main sample: 182 economies (trained and evaluated there). Extended sample: 189 economies (the 7 additions have minor gaps in 1994 or 2024 only).", "",
      "```", T.round(4).to_string(), "```", "", f"Ranking of the models on the main sample: {', '.join(T.sort_values('main_sample_182').index)}.", f"Ranking on the extended sample: {', '.join(T.sort_values('extended_sample_189').index)}.", "",
      "Economies with long gaps (Afghanistan, Syria, Yemen, South Sudan, Eritrea, North Korea, small territories) cannot be added, so the sensitivity covers only part of the sample-selection issue."]
(OUT / "sample_sensitivity.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[4:12]))
