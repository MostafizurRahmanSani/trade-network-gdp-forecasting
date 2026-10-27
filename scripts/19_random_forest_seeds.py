"""Random forest with the same input window as the STGNN (see tabwin.py), five seeds (42, 1, 2, 3, 4) as for the neural models; the configuration is chosen
by the five-seed mean validation MAE over the same grid as 11_baselines_panel_gvar.py. Writes outputs/baselines/RandomForest_win_s<seed>.npz and their mean RandomForest_win5.npz.
(Gradient boosting is deterministic at this sample size: HistGradientBoosting has no subsampling and no early stopping below 10,000 rows.)
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import itertools, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V, tabwin as W
OUT = ROOT / "outputs" / "baselines"
P, iso = V.load(); order = iso.iso3.tolist()
Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}
G = P.pivot(index="iso3", columns="year", values="g").loc[order]
TS = {"train": range(V.FIRST_T, V.TRAIN_END + 1), "val": range(V.TRAIN_END + 1, V.VAL_END + 1), "test": range(V.VAL_END + 1, V.LAST_T + 1)}
X = {k: np.concatenate([W.window(Pw, t) for t in v]) for k, v in TS.items()}; Y = {k: np.concatenate([G[t + 1].values for t in v]) for k, v in TS.items()}
lo, hi = np.percentile(Y["train"], [1, 99]); z = W.scaler(X["train"])
cfg, pr = W.rf_seeds(z(X["train"]), np.clip(Y["train"], lo, hi), z(X["val"]), Y["val"], list(itertools.product((3, 4, 6, 8, 12), (10, 20, 50, 100), (0.3, 0.6, 1.0))), {"test": z(X["test"])})
print("RF config", cfg, flush=True)
for i, s in enumerate(W.SEEDS):
    np.savez(OUT / f"RandomForest_win_s{s}.npz", val_pred=pr["val"][i], val_true=Y["val"], test_pred=pr["test"][i], test_true=Y["test"])
np.savez(OUT / "RandomForest_win5.npz", val_pred=np.mean([pr["val"][i] for i in range(len(W.SEEDS))], axis=0), val_true=np.mean([Y["val"]] * len(W.SEEDS), axis=0), test_pred=np.mean([pr["test"][i] for i in range(len(W.SEEDS))], axis=0), test_true=np.mean([Y["test"]] * len(W.SEEDS), axis=0))
e = np.abs(pr["test"] - Y["test"]).mean(1); print("RF_win 5 seeds: val", np.abs(pr["val"] - Y["val"]).mean(1).round(4), "test", e.round(4), "mean", e.mean().round(4), "sd", e.std().round(5), flush=True)
