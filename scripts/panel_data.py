"""Dataset: windows of yearly country graphs -> next-year real GDP growth.

Samples are indexed by the last input year t (target = growth of year t+1): training t = 2001-2013, validation 2014-2016, test 2017-2023
(target years 2018-2024). Features are winsorised at the 1st/99th percentiles of the training period and standardised with training-period statistics.
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC_EXT = ROOT / "data" / "processed_ext"
WINDOW, FIRST_T, TRAIN_END, VAL_END, LAST_T = 6, 2001, 2013, 2016, 2023
FEATS = {"base": ["g", "loggdp", "logpop", "dpop", "infl", "emp"],
         "ext": ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi"],
         "leannode": ["g", "loggdp", "logpop", "dpop", "partner_im_g"],
         "lean": ["g", "loggdp", "logpop", "dpop", "partner_im_g"],
         "partner": ["g", "loggdp", "logpop", "dpop", "infl", "emp", "gmean", "exp_g", "imp_g", "fuel_share", "hhi", "partner_ex_g", "partner_im_g"]}
EDGE_COLS = ["lv", "xs", "ms"] + [f"gs{k}" for k in range(10)]


def load(proc=None):
    proc = proc or PROC
    P = pd.read_csv(proc / "node_panel.csv.gz"); iso = pd.read_csv(proc / "countries.csv")
    return P, iso


GMODES = ("reversed", "undirected", "randedges", "shufw", "geo", "static")


def _geo():
    import numpy as np, pandas as pd
    iso = pd.read_csv(PROC / "countries.csv"); cc = pd.read_csv(ROOT / "data" / "external" / "country_codes_V202601.csv")
    g = pd.read_csv(ROOT / "data" / "external" / "countries_centroids.csv").drop_duplicates("ISO")
    m = iso.merge(cc[["country_code", "country_iso2"]], left_on="baci", right_on="country_code", how="left").merge(g[["ISO", "longitude", "latitude"]], left_on="country_iso2", right_on="ISO", how="left")
    m.loc[m.iso3 == "HKG", ["longitude", "latitude"]] = (114.17, 22.32); m.loc[m.iso3 == "MAC", ["longitude", "latitude"]] = (113.55, 22.17)
    lon, lat = np.radians(m.longitude.values), np.radians(m.latitude.values)
    a = np.sin((lat[:, None] - lat[None]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


FOLDS = {"gfc": dict(train_end=2005, val_end=2007, last_t=2009),
         "mid": dict(train_end=2009, val_end=2011, last_t=2015),
         "cp": dict(train_end=2011, val_end=2013, cal_end=2016, last_t=2023)}


def build(feats="ext", shuffle_edges=False, seed=0, graph_mode="real", fold=None, norm="global", topk=12, ext=False, horizon=1):
    """Returns dict(train, val, [cal], test, years{split: [t,...]}, iso, clip, g_stats). `fold` selects an alternative split (see FOLDS);
    `norm`='peryear' standardises every feature within each year across countries instead of with training-period statistics.
    `topk` selects the edge files with that many destinations per exporter (12 = default files, 0 = all destinations); `ext` uses the extended-country sensitivity data (targets <= 2023)."""
    sp = dict(train_end=TRAIN_END, val_end=VAL_END, cal_end=None, last_t=LAST_T); sp.update(FOLDS.get(fold, {}) if fold else {})
    TRAIN_END_, VAL_END_, CAL_END_, LAST_T_ = sp["train_end"], sp["val_end"], sp["cal_end"], sp["last_t"]
    proc = PROC_EXT if ext else PROC
    if ext: LAST_T_ = min(LAST_T_, 2022)
    LAST_T_ = min(LAST_T_, 2024 - horizon)
    P, iso = load(proc); codes = iso.baci.tolist(); idx = {c: i for i, c in enumerate(codes)}
    cols = FEATS[feats]
    ecols = EDGE_COLS[:3] if feats == "lean" else EDGE_COLS
    tr_mask = (P.year >= 1996) & (P.year <= TRAIN_END_)
    lo, hi = P.loc[tr_mask, cols].quantile(0.01), P.loc[tr_mask, cols].quantile(0.99)
    Pc = P.copy(); Pc[cols] = Pc[cols].clip(lo, hi, axis=1)
    mu, sd = Pc.loc[tr_mask, cols].mean(), Pc.loc[tr_mask, cols].std().replace(0, 1)
    Pc[cols] = (Pc[cols] - mu) / sd
    if norm == "peryear":
        Pc[cols] = Pc.groupby("year")[cols].transform(lambda c: (c - c.mean()) / (c.std() if c.std() > 0 else 1))
    E = {}
    for y in range(1996, 2024):
        fn = f"edges_{y}.csv.gz" if topk == 12 else f"edges_k{topk if topk else 'all'}_{y}.csv.gz"
        e = pd.read_csv(proc / fn); e["ms"] = np.log10(e.ms + 1e-6); E[y] = e
    tr_e = pd.concat([E[y] for y in range(1996, TRAIN_END_ + 1)])
    em, es = tr_e[["lv", "ms"]].mean(), tr_e[["lv", "ms"]].std()
    if norm == "peryear":
        for y in E: E[y] = E[y].assign(lv=(E[y].lv - E[y].lv.mean()) / E[y].lv.std(), ms=(E[y].ms - E[y].ms.mean()) / E[y].ms.std())
        em, es = pd.Series({"lv": 0.0, "ms": 0.0}), pd.Series({"lv": 1.0, "ms": 1.0})
    gnode = {y: torch.tensor(Pc[Pc.year == y].set_index("iso3").loc[iso.iso3, cols].values, dtype=torch.float32) for y in range(1996, 2025)}
    graph_cache = {}
    D = _geo() if graph_mode == "geo" else None
    def graph(y):
        y_in = y
        if graph_mode == "static":
            y = TRAIN_END_
        if y not in graph_cache:
            e = E[y]
            src = torch.tensor(e.i.map(idx).values); dst = torch.tensor(e.j.map(idx).values)
            if shuffle_edges:
                dst = dst[torch.tensor(np.random.RandomState(seed + y).permutation(len(dst)))]
            ee = e.copy(); ee[["lv", "ms"]] = (ee[["lv", "ms"]] - em) / es
            ea = torch.tensor(ee[ecols].values, dtype=torch.float32); rs = np.random.RandomState(seed + y)
            if graph_mode == "reversed":
                src, dst = dst, src
            elif graph_mode == "undirected":
                src, dst, ea = torch.cat([src, dst]), torch.cat([dst, src]), torch.cat([ea, ea])
            elif graph_mode == "randedges":
                n = len(codes); src = torch.tensor(rs.randint(0, n, len(src))); dst = torch.tensor(rs.randint(0, n, len(dst)))
            elif graph_mode == "shufw":
                ea = ea[torch.tensor(rs.permutation(len(ea)))]
            elif graph_mode == "geo":
                k = int(round(len(src) / len(codes))); Dm = D.copy(); np.fill_diagonal(Dm, np.inf)
                nn_ = np.argsort(Dm, 1)[:, :k]; src = torch.tensor(np.repeat(np.arange(len(codes)), k)); dst = torch.tensor(nn_.ravel())
                ea = torch.zeros(len(src), len(ecols)); ea[:, 0] = torch.tensor((-np.log(Dm[src.numpy(), dst.numpy()])).astype(np.float32)); ea[:, 0] = (ea[:, 0] - ea[:, 0].mean()) / ea[:, 0].std()
            graph_cache[y] = Data(x=gnode[y], edge_index=torch.stack([src, dst]), edge_attr=ea)
        g = graph_cache[y]
        return g if graph_mode != "static" else Data(x=gnode[y_in], edge_index=g.edge_index, edge_attr=g.edge_attr)
    G = P.pivot(index="iso3", columns="year", values="g").loc[iso.iso3]
    lim_lo, lim_hi = np.nanpercentile(G.loc[:, 1996:TRAIN_END_ + horizon].values, [1, 99])
    out = {"train": [], "val": [], "cal": [], "test": [], "years": {"train": [], "val": [], "cal": [], "test": []}, "iso": iso.iso3.tolist(), "clip": (lim_lo, lim_hi), "g_stats": (float(mu["g"]), float(sd["g"]))}
    for t in range(FIRST_T, LAST_T_ + 1):
        gs = [graph(y) for y in range(t - WINDOW + 1, t + 1)]
        y = torch.tensor(G[t + horizon].values.astype(np.float32)).unsqueeze(1)
        split = "train" if t <= TRAIN_END_ else "val" if t <= VAL_END_ else "cal" if (CAL_END_ and t <= CAL_END_) else "test"
        if split == "train": y = y.clamp(lim_lo, lim_hi)
        out[split].append((gs, y)); out["years"][split].append(t + horizon)
    return out


if __name__ == "__main__":
    d = build("ext")
    print({k: len(d[k]) for k in ("train", "val", "test")}, d["years"], d["clip"])
    g = d["train"][0][0][0]; print(g)
