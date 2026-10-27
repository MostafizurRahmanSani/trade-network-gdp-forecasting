"""Edge files with a different number of destinations per exporter (k = 5, 25, all) and full-network statistics.

Same edge features as 02_build_panel.py. Writes data/processed/edges_k{k}_{year}.csv.gz and outputs/graph_density_full_network.md (directed density of the
complete BACI network among the 182 economies, and the export coverage of the top-k graph).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INT, OUT = ROOT / "data" / "interim", ROOT / "data" / "processed"
codes = pd.read_csv(OUT / "countries.csv").baci.tolist(); n = len(codes)
cc = pd.read_csv(ROOT / "data" / "external" / "country_codes_V202601.csv")
_cur = dict(zip(cc.country_iso3[~cc.country_name.str.contains(r"\(\.\.\.", regex=True)], cc.country_code[~cc.country_name.str.contains(r"\(\.\.\.", regex=True)]))
alias = {int(r.country_code): int(_cur[r.country_iso3]) for r in cc.itertuples() if r.country_iso3 in _cur and int(r.country_code) != int(_cur[r.country_iso3])}
rows = []
for y in range(1996, 2024):
    t = pd.read_csv(INT / f"pair_group_{y}.csv.gz"); t["i"], t["j"] = t.i.replace(alias), t.j.replace(alias); t = t.groupby(["i", "j", "g"], as_index=False).v.sum()
    t = t[t.i.isin(codes) & t.j.isin(codes) & (t.i != t.j)]
    pairs = t.groupby(["i", "j"]).v.sum().rename("v").reset_index()
    EX = t.groupby("i").v.sum().reindex(codes).fillna(0); IM = t.groupby("j").v.sum().reindex(codes).fillna(0)
    srt = pairs.sort_values(["i", "v"], ascending=[True, False])
    stat = dict(year=y, full_edges=len(pairs), full_density=len(pairs) / (n * (n - 1)))
    for k in (5, 12, 25, 0):
        a = srt.groupby("i").head(k) if k else srt.copy()
        cover = a.groupby("i").v.sum() / srt.groupby("i").v.sum()
        stat[f"cover_median_k{k or 'all'}"] = float(cover.median()); stat[f"edges_k{k or 'all'}"] = len(a)
        if k in (5, 25, 0):
            a = a.copy(); a["xs"] = a.v / a.i.map(EX.set_axis(codes)).clip(lower=1); a["ms"] = a.v / a.j.map(IM.set_axis(codes)).clip(lower=1)
            gs = t.merge(a[["i", "j"]], on=["i", "j"]).pivot_table(index=["i", "j"], columns="g", values="v", fill_value=0)
            gs = gs.div(gs.sum(1), axis=0).reindex(columns=range(10), fill_value=0).add_prefix("gs").reset_index()
            a = a.merge(gs, on=["i", "j"], how="left").fillna(0); a["lv"] = np.log1p(a.v)
            a.to_csv(OUT / f"edges_k{k or 'all'}_{y}.csv.gz", index=False, compression="gzip")
    rows.append(stat); print(y, stat["full_density"], flush=True)
S = pd.DataFrame(rows); S.to_csv(ROOT / "outputs" / "graph_density_full_network.csv", index=False)
md = ["# Full trade network vs the top-k graphs used by the model (182 countries)", "",
      "Directed density of the complete BACI network (all exporter-importer pairs with positive trade, all products summed) and the median share of an exporter's total exports captured by its top-k destinations.", "",
      "```", S[["year", "full_edges", "full_density", "cover_median_k5", "cover_median_k12", "cover_median_k25", "edges_k5", "edges_k12", "edges_k25"]].iloc[[0, 4, 9, 14, 19, 23, 27]].round(3).to_string(index=False), "```", "",
      f"The complete network is dense (directed density {S.full_density.iloc[0]:.2f} in 1996, {S.full_density.iloc[-5:].mean():.2f} in 2019-2023); the model graph (k = 12) has density about 0.065 only because each exporter keeps its 12 largest destinations, which carry a median {S.cover_median_k12.mean():.0%} of its exports."]
(ROOT / "outputs" / "graph_density_full_network.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md[-4:]))
