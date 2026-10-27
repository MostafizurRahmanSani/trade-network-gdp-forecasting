"""Report of the extended tuning check. Rule fixed in advance: a configuration from the extended grid replaces the selected one only if its five-seed mean
validation MAE is lower by more than 0.0003 (about 1.5%, larger than the seed noise of the validation MAE). Writes outputs/tuning_decisions.json.
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import selection
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
FAM = ["STGNN", "STGCN", "STGAT", "DCRNN", "GWN", "GNN", "GCN", "GAT", "GRU", "LSTM", "RNN", "BiLSTM", "Transformer", "TFT", "MTGNN", "MTGNN_trade"]
THR = 0.0003
md = ["# Extended tuning check (neural families)", "", f"Around each family's selected configuration (feature set, hidden, weight decay from the original grid) I added learning rate 3e-4 and 3e-3 (selected: 1e-3), "
      "hidden size 8 (families selected with 16) or 64 (families selected with 32) and weight decay 1e-2. Each extra configuration: 5 seeds, validation MAE; test results play no role.",
      f"Decision rule (fixed beforehand): adopt an extra configuration only if its mean validation MAE beats the selected one by more than {THR}.", "",
      "| Family | selected (feat, h, wd) | val | lr 3e-4 | lr 3e-3 | hidden 8/64 | wd 1e-2 | best extra | improvement | adopt |", "|---|---|---|---|---|---|---|---|---|---|"]
dec = {}
for m in FAM:
    sel = selection.selected(m)
    if not sel: continue
    f, h, wd = sel; v0 = selection._val(m, sel, (42, 1, 2, 3, 4))
    def v(name, hh, ww):
        fs = [RUNS / f"{name}_{f}_h{hh}_wd{ww}_s{s}.json" for s in (42, 1, 2, 3, 4)]
        return float(np.mean([json.load(open(x))["val_MAE"] for x in fs])) if all(x.exists() for x in fs) else None
    cand = {"lr 3e-4": (f"{m}#lr0.0003", h, wd), "lr 3e-3": (f"{m}#lr0.003", h, wd), f"hidden {8 if h == 16 else 64}": (m, 8 if h == 16 else 64, wd), "wd 1e-2": (m, h, "1e-2")}
    vals = {k: v(*c) for k, c in cand.items()}
    ok = {k: x for k, x in vals.items() if x is not None}
    bk = min(ok, key=ok.get) if ok else None; imp = (v0 - ok[bk]) if bk else 0
    adopt = bool(bk and imp > THR)
    if adopt: dec[m] = dict(name=cand[bk][0], feats=f, h=cand[bk][1], wd=cand[bk][2], improvement=imp)
    cell = lambda k: "n/a" if vals.get(k) is None else f"{vals[k]:.5f}"
    md.append(f"| {m} | {f}, {h}, {wd} | {v0:.5f} | {cell('lr 3e-4')} | {cell('lr 3e-3')} | {cell(f'hidden {8 if h == 16 else 64}')} | {cell('wd 1e-2')} | {bk or '-'} | {imp:+.5f} | {'**yes**' if adopt else 'no'} |")
md += ["", f"Families adopting a new configuration: {', '.join(dec) if dec else 'none'}."]
(OUT / "tuning_extension.md").write_text("\n".join(md), encoding="utf-8"); json.dump(dec, open(OUT / "tuning_decisions.json", "w"), indent=1); print("\n".join(md[6:]))
