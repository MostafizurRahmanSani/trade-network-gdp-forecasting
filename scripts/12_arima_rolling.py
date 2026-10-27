"""Rolling one-step ARIMA on log real GDP: refit at every forecast origin on an expanding window (1996..t), per country, with one order common to all
countries chosen on validation MAE among five candidates. Growth forecast = forecast(log GDP t+1) - log GDP t.
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
L = P.pivot(index="iso3", columns="year", values="loggdp").loc[order]
G = L.diff(axis=1)
VAL, TEST = list(range(V.TRAIN_END + 1, V.VAL_END + 1)), list(range(V.VAL_END + 1, V.LAST_T + 1))


def preds(o, origins):
    out, tru = [], []
    for t in origins:
        for c in order:
            s = L.loc[c, 1996:t].values
            try:
                out.append(ARIMA(s, order=o).fit(method_kwargs={"maxiter": 200}).forecast(1)[0] - s[-1])
            except Exception:
                out.append(np.mean(np.diff(s)))
        tru.extend(G[t + 1].values)
    return np.array(out), np.array(tru)


best = None
for o in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 1, 0), (0, 1, 1)]:
    t0 = time.time(); vp, vt = preds(o, VAL); mae = np.abs(vp - vt).mean()
    print(f"ARIMA{o} val MAE {mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if best is None or mae < best[0]: best = (mae, o, vp, vt)
tp, tt = preds(best[1], TEST)
yr = np.repeat(range(2018, 2025), len(order)); cov = np.isin(yr, (2020, 2021)); e = np.abs(tp - tt)
np.savez(ROOT / "outputs" / "baselines" / "ARIMA_rolling.npz", val_pred=best[2], val_true=best[3], test_pred=tp, test_true=tt)
print(f"SELECTED {best[1]} val {best[0]:.4f} | test {e.mean():.4f} exCOVID {e[~cov].mean():.4f} COVID {e[cov].mean():.4f}")
print("per-year", [round(float(e[yr == t].mean()), 4) for t in range(2018, 2025)])
