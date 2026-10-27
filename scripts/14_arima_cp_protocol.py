"""ARIMA (without drift) for the conformal protocol: the order is selected only on the early-stopping years 2013-14 (origins 2012-13), so that the
calibration years 2015-17 are not used for any selection. Same procedure as 12_arima_rolling.py otherwise.
"""
import os, sys, warnings, time
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
P, iso = V.load(); order = iso.iso3.tolist()
L = P.pivot(index="iso3", columns="year", values="loggdp").loc[order]; G = L.diff(axis=1)


def preds(o, origins):
    out = []
    for t in origins:
        for c in order:
            s = L.loc[c, 1996:t].values
            try: out.append(ARIMA(s, order=o).fit(method_kwargs={"maxiter": 200}).forecast(1)[0] - s[-1])
            except Exception: out.append(np.mean(np.diff(s)))
    return np.array(out)


VAL, CAL, TEST = [2012, 2013], [2014, 2015, 2016], list(range(2017, 2024))
yv = np.concatenate([G[t + 1].values for t in VAL])
best = None
for o in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 1, 0), (0, 1, 1)]:
    e = np.abs(preds(o, VAL) - yv).mean(); print(f"ARIMA{o} val(2013-14) MAE {e:.4f}", flush=True)
    if best is None or e < best[0]: best = (e, o)
o = best[1]; print("selected", o)
np.savez(ROOT / "outputs" / "baselines" / "ARIMA_cp.npz", val_pred=preds(o, VAL), val_true=yv, cal_pred=preds(o, CAL), cal_true=np.concatenate([G[t + 1].values for t in CAL]),
         test_pred=preds(o, TEST), test_true=np.concatenate([G[t + 1].values for t in TEST]), order=np.array(o))
print("saved")
