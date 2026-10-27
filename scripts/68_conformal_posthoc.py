"""Conformal analyses that need only saved point forecasts (no model is trained or fitted).

Point forecasts: conformal-protocol runs of the STGNN, GRU and LSTM (five-seed means), the conformal-protocol rolling ARIMA (baselines/ARIMA_cp_drift.npz), and two naive
forecasts that need no fitting: no-change (zero growth) and random walk with drift. Writes outputs/conformal_posthoc.md.
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_data as V
OUT = ROOT / "outputs"; RUNS = OUT / "runs"
P, iso = V.load(); order = iso.iso3.tolist(); N = len(order)
G = P.pivot(index="iso3", columns="year", values="g").loc[order]
TS = {"cal": range(2014, 2017), "test": range(2017, 2024)}
Y = {k: np.concatenate([G[t + 1].values for t in v]) for k, v in TS.items()}
YRS = list(range(2018, 2025)); yr = np.repeat(YRS, N)


def ens(prefix):
    fs = [f for f in sorted(glob.glob(str(RUNS / f"{prefix}_s*.npz"))) if f.rsplit("_s", 1)[1][:-4].isdigit()]
    D = [np.load(f) for f in fs]; assert len(D) == 5, prefix
    return {"cal": np.mean([d["cal_pred"] for d in D], 0), "test": np.mean([d["test_pred"] for d in D], 0), "es": np.mean([np.abs(d["val_pred"] - d["val_true"]).mean() for d in D])}


pts = {"STGNN": ens("STGNN@cp_base_h32_wd1e-3"), "LSTM": ens("LSTM@cp_base_h16_wd1e-3"), "GRU": ens("GRU@cp_base_h16_wd1e-3")}
ar = np.load(OUT / "baselines" / "ARIMA_cp_drift.npz"); pts["Rolling ARIMA"] = {"cal": ar["cal_pred"], "test": ar["test_pred"]}
pts["No-change (naive)"] = {"cal": np.zeros(len(Y["cal"])), "test": np.zeros(len(Y["test"]))}
pts["RW with drift (naive)"] = {k: np.concatenate([np.nan_to_num(G.loc[:, 1996:t].mean(axis=1).values) for t in TS[k]]) for k in TS}

vol = G.loc[:, 1996:2017].std(axis=1).values; vt = np.digitize(vol, np.quantile(vol, [1 / 3, 2 / 3]))
gcal, gtest = np.tile(vt, 3), np.tile(vt, 7)


def conf_q(s, alpha):
    n = len(s); return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def winkler(l, h, y, a):
    return (h - l) + (2 / a) * (l - y) * (y < l) + (2 / a) * (y - h) * (y > h)


def run(name, method, alpha, gamma=0.05, corrected=False):
    pc, pt = pts[name]["cal"], pts[name]["test"]; sc = np.abs(Y["cal"] - pc); yt = Y["test"]; l = np.zeros_like(pt); h = np.zeros_like(pt)
    if method == "split":
        q = conf_q(sc, alpha); return pt - q, pt + q
    if method == "mondrian":
        for g in range(3):
            q = conf_q(sc[gcal == g], alpha); l[gtest == g], h[gtest == g] = pt[gtest == g] - q, pt[gtest == g] + q
        return l, h
    pool = [sc]; a_t = alpha
    for i in range(7):
        sl = slice(i * N, (i + 1) * N); s_all = np.concatenate(pool)
        if method == "expanding":
            q = conf_q(s_all, alpha)
        else:
            q = conf_q(s_all, a_t) if corrected else np.quantile(s_all, 1 - a_t, method="higher")
        l[sl], h[sl] = pt[sl] - q, pt[sl] + q
        s_t = np.abs(yt[sl] - pt[sl]); pool.append(s_t)
        a_t = np.clip(a_t + gamma * (alpha - np.mean(s_t > q)), 0.001, 0.999)
    return l, h


rng = np.random.RandomState(0); BIDX = rng.randint(0, 7, (5000, 7))


def summary(l, h, alpha):
    y = Y["test"]; c = (y >= l) & (y <= h); w = winkler(l, h, y, alpha)
    cy = np.array([c[yr == t].mean() for t in YRS]); wy = np.array([w[yr == t].mean() for t in YRS])
    bc, bw = cy[BIDX].mean(1), wy[BIDX].mean(1); ex = ~np.isin(yr, (2020, 2021))
    return dict(cov=c.mean(), cov_lo=np.percentile(bc, 5), cov_hi=np.percentile(bc, 95), cov_ex2021=c[ex].mean(), cov2020=cy[2], cov2021=cy[3],
                width=float(np.mean(h - l)), WS=w.mean(), WS_lo=np.percentile(bw, 5), WS_hi=np.percentile(bw, 95), WS_ex2021=w[ex].mean()), wy, c


md = ["# Conformal post-hoc analyses on saved CP-protocol forecasts (no refitting)", "",
      "CP protocol: training targets <= 2012, early stopping 2013-2014, calibration 2015-2017, test 2018-2024. Bootstrap: 5,000 resamples of the 7 test years (90% percentile interval).", "",
      "## A. Point accuracy of the CP-protocol forecasts", "", "| Model | early-stopping MAE 2013-14 | calibration MAE 2015-17 | test MAE 2018-24 |", "|---|---|---|---|"]
for k, v in pts.items():
    md.append(f"| {k} | {v.get('es', float('nan')):.5f} | {np.abs(v['cal'] - Y['cal']).mean():.5f} | {np.abs(v['test'] - Y['test']).mean():.5f} |")
WY = {}
for alpha in (0.10, 0.05):
    md += ["", f"## B. Nominal {1 - alpha:.0%}", "", "| Model | method | coverage [90% CI] | cov. excl. 2020-21 | cov. 2020 | cov. 2021 | width | Winkler [90% CI] | WS excl. 2020-21 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in pts:
        for meth, kw, lab in (("split", {}, "split CP"), ("mondrian", {}, "Mondrian split CP (volatility terciles)"), ("expanding", {}, "expanding-window CP (no alpha update)"),
                              ("aci", dict(gamma=0.05), "ACI gamma 0.05 (as in paper)"), ("aci", dict(gamma=0.01), "ACI gamma 0.01"), ("aci", dict(gamma=0.10), "ACI gamma 0.10"),
                              ("aci", dict(gamma=0.05, corrected=True), "ACI gamma 0.05, corrected quantile")):
            l, h = run(k, meth, alpha, **kw); s, wy, c = summary(l, h, alpha); WY[(k, lab, alpha)] = wy
            md.append(f"| {k} | {lab} | {s['cov']:.3f} [{s['cov_lo']:.3f}, {s['cov_hi']:.3f}] | {s['cov_ex2021']:.3f} | {s['cov2020']:.3f} | {s['cov2021']:.3f} | {s['width']:.3f} | {s['WS']:.3f} [{s['WS_lo']:.3f}, {s['WS_hi']:.3f}] | {s['WS_ex2021']:.3f} |")
            if meth == "mondrian" and alpha == 0.10:
                md.append(f"|  | Mondrian coverage by volatility tercile (low/medium/high) | " + " / ".join(f"{c[gtest == g].mean():.3f}" for g in range(3)) + " | | | | | | |")
            if meth == "split" and alpha == 0.10:
                md.append(f"|  | split coverage by volatility tercile (low/medium/high) | " + " / ".join(f"{c[gtest == g].mean():.3f}" for g in range(3)) + " | | | | | | |")
md += ["", "## C. Year-clustered DM test on the Winkler loss (split CP, 90%; negative = STGNN better; HLN, t(6))", "", "| Comparison | mean diff | p | years STGNN better |", "|---|---|---|---|"]
for k in pts:
    if k == "STGNN": continue
    d = WY[("STGNN", "split CP", 0.10)] - WY[(k, "split CP", 0.10)]; hln = d.mean() / (d.std(ddof=1) / np.sqrt(7)) * np.sqrt(6 / 7)
    md.append(f"| STGNN vs {k} | {d.mean():+.4f} | {2 * stats.t.sf(abs(hln), 6):.3f} | {int((d < 0).sum())} |")
d = WY[("STGNN", "ACI gamma 0.05 (as in paper)", 0.10)] - WY[("STGNN", "split CP", 0.10)]; hln = d.mean() / (d.std(ddof=1) / np.sqrt(7)) * np.sqrt(6 / 7)
md += [f"| STGNN: ACI vs split CP | {d.mean():+.4f} | {2 * stats.t.sf(abs(hln), 6):.3f} | {int((d < 0).sum())} (ACI better) |", ""]
pc, pt = pts["STGNN"]["cal"], pts["STGNN"]["test"]; a_t = 0.10; path = []
pool = [np.abs(Y["cal"] - pc)]
for i in range(7):
    sl = slice(i * N, (i + 1) * N); q = np.quantile(np.concatenate(pool), 1 - a_t, method="higher"); s_t = np.abs(Y["test"][sl] - pt[sl])
    path.append((YRS[i], a_t, q, np.mean(s_t <= q))); pool.append(s_t); a_t = np.clip(a_t + 0.05 * (0.10 - np.mean(s_t > q)), 0.001, 0.999)
md += ["## D. ACI path for the STGNN at 90% (alpha used in each test year, q, coverage)", "", "| year | alpha_t | q | coverage |", "|---|---|---|---|"] + [f"| {a} | {b:.4f} | {c:.4f} | {d:.3f} |" for a, b, c, d in path]
(OUT / "conformal_posthoc.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
