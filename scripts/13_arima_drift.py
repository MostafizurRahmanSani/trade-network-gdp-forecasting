"""Rolling one-step ARIMA(p,1,q) on log real GDP with a drift term (statsmodels trend='t': a constant in the differenced series), otherwise identical to
12_arima_rolling.py: refit at every origin on an expanding window, one order for all countries chosen on validation MAE (targets 2015-17).
With the argument 'cp' it produces the conformal-protocol version (order chosen on targets 2013-14 only; calibration 2015-17; test 2018-24).
Outputs: outputs/baselines/ARIMA_rolling_drift.npz or ARIMA_cp_drift.npz.
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
ORDERS = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 1, 0), (0, 1, 1)]


def preds(o, origins):
    out, fails = [], 0
    for t in origins:
        for c in order:
            s = L.loc[c, 1996:t].values
            try:
                out.append(ARIMA(s, order=o, trend="t").fit(method_kwargs={"maxiter": 200}).forecast(1)[0] - s[-1])
            except Exception:
                out.append(np.mean(np.diff(s))); fails += 1
    return np.array(out), fails


cp = len(sys.argv) > 1 and sys.argv[1] == "cp"
if cp:
    VAL, CAL, TEST = [2012, 2013], [2014, 2015, 2016], list(range(2017, 2024))
else:
    VAL, CAL, TEST = list(range(V.TRAIN_END + 1, V.VAL_END + 1)), [], list(range(V.VAL_END + 1, V.LAST_T + 1))
yv = np.concatenate([G[t + 1].values for t in VAL])
best = None
for o in ORDERS:
    t0 = time.time(); vp, f = preds(o, VAL); e = np.abs(vp - yv).mean()
    print(f"ARIMA{o}+drift val MAE {e:.5f} (fallbacks {f}, {time.time() - t0:.0f}s)", flush=True)
    if best is None or e < best[0]: best = (e, o, vp)
o = best[1]; tp, f = preds(o, TEST); tt = np.concatenate([G[t + 1].values for t in TEST])
res = dict(val_pred=best[2], val_true=yv, test_pred=tp, test_true=tt, order=np.array(o))
if cp:
    cpred, _ = preds(o, CAL); res.update(cal_pred=cpred, cal_true=np.concatenate([G[t + 1].values for t in CAL]))
fn = ROOT / "outputs" / "baselines" / ("ARIMA_cp_drift.npz" if cp else "ARIMA_rolling_drift.npz")
assert not fn.exists(), f"{fn} exists; not overwriting"
np.savez(fn, **res)
yr = np.repeat(range(2018, 2025), len(order)); cov = np.isin(yr, (2020, 2021)); e = np.abs(tp - tt)
print(f"SELECTED {o} val {best[0]:.5f} | test {e.mean():.5f} exCOVID {e[~cov].mean():.5f} COVID {e[cov].mean():.5f} | test fallbacks {f}", flush=True)
