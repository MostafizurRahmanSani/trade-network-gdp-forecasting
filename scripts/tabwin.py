"""Helpers for the tabular baselines that receive the same input window as the STGNN: the six node features of years t-5..t flattened into 36 inputs.
The random forest is averaged over the same five seeds as the neural models.
"""
import itertools
import numpy as np
from sklearn.ensemble import RandomForestRegressor

BASE = ["g", "loggdp", "logpop", "dpop", "infl", "emp"]
SEEDS = (42, 1, 2, 3, 4)


def window(Pw, t):
    return np.column_stack([Pw[t - j][c].values for j in range(6) for c in BASE])


def scaler(Xtr):
    a, b = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0); c = np.clip(Xtr, a, b); mu, sd = c.mean(0), c.std(0) + 1e-9
    return lambda X: np.nan_to_num((np.clip(X, a, b) - mu) / sd)


def rf_seeds(Ztr, ytr, Zv, yv, grid, others, n_sel=150, n_final=300):
    """Select (depth, leaf, max_features) by the 5-seed mean validation MAE; refit each seed with n_final trees.
    Returns (config, {name: array (5, n)}) for Zv ('val') and every matrix in `others`."""
    best = min(grid, key=lambda cfg: np.mean([np.abs(RandomForestRegressor(n_estimators=n_sel, max_depth=cfg[0], min_samples_leaf=cfg[1], max_features=cfg[2], n_jobs=1,
                                                                             random_state=s).fit(Ztr, ytr).predict(Zv) - yv).mean() for s in SEEDS]))
    out = {"val": [], **{k: [] for k in others}}
    for s in SEEDS:
        m = RandomForestRegressor(n_estimators=n_final, max_depth=best[0], min_samples_leaf=best[1], max_features=best[2], n_jobs=1, random_state=s).fit(Ztr, ytr)
        out["val"].append(m.predict(Zv))
        for k, Z in others.items(): out[k].append(m.predict(Z))
    return best, {k: np.array(v) for k, v in out.items()}
