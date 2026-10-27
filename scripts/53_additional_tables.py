"""Stability and checkpoint study table, per-year-normalisation check, capacity check and reproducibility table (parameter counts, run times, selected configurations)."""
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
md = ["# Additional tables ", ""]


def mae_seeds(pat):
    fs = [f for f in sorted(glob.glob(str(RUNS / f"{pat}_s*.json"))) if f.rsplit("_s", 1)[1][:-5].isdigit()]
    if not fs: return None
    J = [json.load(open(f)) for f in fs]; D = [np.load(f.replace(".json", ".npz")) for f in fs]
    return len(fs), np.mean([j["val_MAE"] for j in J]), np.mean([j["test_MAE"] for j in J]), np.std([j["test_MAE"] for j in J]), np.mean([j["sec"] for j in J])


S = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(OUT / "stability" / "*.json"))])
if len(S):
    g = S.groupby("config")
    T = pd.DataFrame({"seeds": g.size(), "val(best epoch)": g.val_best.mean(), "test @ best-val epoch": g.test_at_best_val.mean(), "test mean over +-5 epochs": g.test_mean_pm5_around_best.mean(),
                      "test last-10 weight avg": g.test_lastK_avgweights.mean(), "test last epoch": g.test_last_epoch.mean(), "sd of test MAE over last 50 epochs": g.test_sd_last50.mean(),
                      "sd of val MAE over last 50 epochs": g.val_sd_last50.mean(), "mean best epoch": g.best_epoch.mean(), "sd test@best across seeds": g.test_at_best_val.std()})
    md += ["## 1. Training stability and checkpoint sensitivity (STGNN, base features, h16, wd 1e-3, 5 seeds)", "",
           "Configurations: epochs 100/300/1000, gradient clipping (max-norm 1), cosine learning-rate schedule. Checkpoint rules: best-validation epoch, mean over +-5 epochs around it, average of the weights of the last 10 epochs, last epoch.", "",
           "```", T.round(5).loc[[c for c in ("e100", "e300", "e300_clip", "e300_cos", "e300_clip_cos", "e1000", "e1000_clip_cos") if c in T.index]].to_string(), "```", ""]

md += ["## 2. Node/edge feature normalisation: training-period (global) vs per-year (5 seeds each, same configuration)", "", "| Model | normalisation | seeds | val MAE | test MAE | sd |", "|---|---|---|---|---|---|"]
import selection
for m in ("STGNN", "LSTM", "GRU"):
    rn, f, h, wd = selection.final_config(m)
    for lab, pat in (("training-period (used)", f"{rn}_{f}_h{h}_wd{wd}"), ("per-year", f"{selection.compose(rn, 'pynorm')}_{f}_h{h}_wd{wd}")):
        r = mae_seeds(pat)
        if r: md.append(f"| {m} | {lab} | {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} |")
md += [""]

from gdp_stgnn.models import GRU, LSTM
import train_neural; train_neural.EDGE_DIM = 13
cnt = lambda m: sum(p.numel() for p in m.parameters())
FN = {"base": 6, "ext": 11, "partner": 13, "lean": 5, "leannode": 5}
md += ["## 3. Capacity check: recurrent baselines with more parameters than the STGNN", "", "| Model | hidden | parameters | seeds | val MAE | test MAE |", "|---|---|---|---|---|---|"]
for m in ("LSTM", "GRU"):
    rn, f, h0, wd = selection.final_config(m)
    for h in sorted({h0, 48, 64}):
        r = mae_seeds(f"{rn}_{f}_h{h}_wd{wd}")
        if r: md.append(f"| {m} ({f}) | {h} | {cnt(train_neural.build_model(m, FN[f], h))} | {r[0]} | {r[1]:.4f} | {r[2]:.4f} |")
rn, f, h, wd = selection.final_config("STGNN"); r = mae_seeds(f"{rn}_{f}_h{h}_wd{wd}"); md.append(f"| STGNN ({f}) | {h} | {cnt(train_neural.build_model('STGNN', FN[f], h))} | {r[0]} | {r[1]:.4f} | {r[2]:.4f} |")
md += [""]

FEATN = {"base": 6, "ext": 11, "partner": 13, "lean": 5, "leannode": 5}
md += ["## 4. Parameter counts, run time and selected configuration (validation-selected, 5 seeds)", "", "| Model | features | hidden | wd | parameters | seconds per run (CPU, 1 thread) |", "|---|---|---|---|---|---|"]
fam = ["STGNN", "STGCN", "STGAT", "DCRNN", "GWN", "GNN", "GCN", "GAT", "BonnigGAT", "BonnigGAT_edges", "GRU", "LSTM", "RNN", "BiLSTM", "Transformer", "TFT", "MTGNN", "MTGNN_trade"]
for m in fam:
    fc = selection.final_config(m)
    if not fc: continue
    rn, f, h, wd = fc
    fs = [RUNS / f"{rn}_{f}_h{h}_wd{wd}_s{s}.json" for s in (42, 1, 2, 3, 4)]; fs = [x for x in fs if x.exists()]
    if not fs: continue
    sec = np.mean([json.load(open(x))["sec"] for x in fs]); lr = f", lr {rn.split('#lr')[1]}" if "#lr" in rn else ""
    md.append(f"| {m} | {f} ({FEATN[f]}){lr} | {h} | {wd} | {cnt(train_neural.build_model(m, FEATN[f], int(h)))} | {sec:.0f} |")
md += ["", "Selection protocol (every neural family, validation MAE only): all models use the 6 base node features (the larger sets {ext: 11, partner: 13} were part of the first grid and are out of the final selection by author decision); "
       "candidate configurations hidden size {16, 32} x weight decay {0, 1e-3}, each with 5 seeds (42, 1, 2, 3, 4); the configuration with the lowest 5-seed mean validation MAE is selected; "
       "an extended check (learning rate 3e-4/3e-3, hidden size 8 or 64, weight decay 1e-2; rule: adopt only if validation improves by more than 0.0003) changed only the LSTM (hidden 8). "
       "Adam, learning rate 1e-3, up to 300 epochs, checkpoint with the lowest validation MAE. The seed controls weight initialisation, the mini-batch order of the recurrent baselines and the randomised graph controls; "
       "the spatio-temporal models visit training windows in chronological order; tabular baselines use random_state=0.",
       "Message direction: edges run exporter -> importer (message from source i to destination j); spatial-only models (GNN, GCN, GAT, Bonnig-style GAT) use the graph of the last input year."]
(OUT / "additional_tables.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[:60]))
