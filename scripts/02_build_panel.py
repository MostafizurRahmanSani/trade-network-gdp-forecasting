"""Clean and assemble the node panel and the edge tables from the BACI aggregates and the World Bank (WDI) extract.

Outputs (data/processed): countries.csv, node_panel.csv.gz (country-year features), edges_{year}.csv.gz (top-12 export destinations per exporter, 13 edge
features), cleaning_report.md.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INT, OUT = ROOT / "data" / "interim", ROOT / "data" / "processed"
Y0, Y1, TOPK = 1995, 2024, 12
rep = []


def load_wdi():
    d = pd.read_csv(RAW / "wdi" / "WDI_extract.csv", na_values="..")
    d = d[d["Series Code"].notna() & d["Country Code"].notna()]
    d.columns = [c.split(" ")[0] if "[YR" in c else c for c in d.columns]
    names = {"NY.GDP.MKTP.KD": "gdp", "SP.POP.TOTL": "pop", "FP.CPI.TOTL.ZG": "infl", "SL.EMP.TOTL.SP.ZS": "emp"}
    out = {}
    for code, nm in names.items():
        s = d[d["Series Code"] == code].set_index("Country Code")[[str(y) for y in range(1990, 2025)]]
        s.columns = range(1990, 2025); out[nm] = s.astype(float)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wdi = load_wdi()
    cc = pd.read_csv(ROOT / "data" / "external" / "country_codes_V202601.csv")
    code2iso = dict(zip(cc.country_code, cc.country_iso3))
    gdp, pop = wdi["gdp"], wdi["pop"]
    ok = [c for c in gdp.index if c in set(cc.country_iso3)
          and gdp.loc[c, 1994:2024].notna().all() and pop.loc[c, 1994:2024].notna().all()]
    dropped = sorted(set(gdp.index) - set(ok))
    rep.append(f"- WDI rows: {len(gdp)}; kept {len(ok)} economies present in BACI with complete real GDP & population 1994-2024.")
    rep.append(f"- Dropped (aggregates or incomplete GDP/pop): {len(dropped)}")
    obsolete = cc.country_name.str.contains(r"\(\.\.\.", regex=True)
    cur = dict(zip(cc.country_iso3[~obsolete], cc.country_code[~obsolete]))
    alias = {int(r.country_code): int(cur[r.country_iso3]) for r in cc.itertuples() if r.country_iso3 in cur and int(r.country_code) != int(cur[r.country_iso3])}
    rep.append(f"- Obsolete BACI codes merged into the current code of the same economy: {alias}")
    iso2code = {v: int(k) for v, k in cur.items() if v in ok}
    ok = [c for c in ok if c in iso2code]; codes = [iso2code[c] for c in ok]
    pd.DataFrame({"iso3": ok, "baci": codes}).to_csv(OUT / "countries.csv", index=False)

    tot_ex, tot_im, fuel_sh, hhi, pair = {}, {}, {}, {}, {}
    for y in range(Y0, Y1 + 1):
        t = pd.read_csv(INT / f"pair_group_{y}.csv.gz")
        t["i"], t["j"] = t.i.replace(alias), t.j.replace(alias)
        t = t.groupby(["i", "j", "g"], as_index=False).v.sum()
        t = t[t.i.isin(codes) & t.j.isin(codes) & (t.i != t.j)]
        pair[y] = t
        ex = t.groupby("i").v.sum().reindex(codes).fillna(0); im = t.groupby("j").v.sum().reindex(codes).fillna(0)
        tot_ex[y], tot_im[y] = ex, im
        gx = t.groupby(["i", "g"]).v.sum().unstack(fill_value=0).reindex(codes).fillna(0)
        sh = gx.div(gx.sum(1).replace(0, np.nan), axis=0)
        fuel_sh[y] = sh.get(1, pd.Series(0, index=codes)).fillna(0); hhi[y] = (sh ** 2).sum(1).fillna(0)
    EX, IM = pd.DataFrame(tot_ex), pd.DataFrame(tot_im); EX.index = IM.index = ok

    rows = {}
    gdp, pop = np.log(gdp.loc[ok, 1994:2024]), np.log(pop.loc[ok, 1994:2024])
    infl = wdi["infl"].loc[ok, 1994:2024]; emp = wdi["emp"].loc[ok, 1994:2024]
    miss_infl, miss_emp = infl.isna().mean(1), emp.isna().mean(1)
    rep.append(f"- Inflation missing cells: {int(infl.isna().sum().sum())} ; employment-ratio missing cells: {int(emp.isna().sum().sum())}")
    for nm, df in (("infl", infl), ("emp", emp)):
        df[:] = df.interpolate(axis=1, limit=3, limit_direction="both")
        n_left = int(df.isna().sum().sum())
        df[:] = df.apply(lambda c: c.fillna(c.median()), axis=0)
        rep.append(f"- {nm}: {n_left} cells left after <=3y interpolation -> imputed with the cross-country median of that year")
    lo, hi = np.nanpercentile(infl.values, [1, 99])
    rep.append(f"- Inflation winsorised to [{lo:.1f}, {hi:.1f}]% (1st/99th percentile) then signed-log1p transformed")
    infl = np.sign(infl.clip(lo, hi)) * np.log1p(infl.clip(lo, hi).abs() / 100)
    g = gdp.diff(axis=1)
    lex, lim = np.log(EX.clip(lower=1)), np.log(IM.clip(lower=1))
    pex, pim = {}, {}
    for y in range(Y0 + 1, Y1 + 1):
        t = pair[y - 1]; gv = g[y].reindex(ok).fillna(0)
        gi = dict(zip(codes, gv.values))
        a = t.groupby(["i", "j"]).v.sum().reset_index(); a["gj"] = a.j.map(gi); a["gi"] = a.i.map(gi)
        pex[y] = (a.assign(w=a.v * a.gj).groupby("i").w.sum() / a.groupby("i").v.sum()).reindex(codes).fillna(0).values
        pim[y] = (a.assign(w=a.v * a.gi).groupby("j").w.sum() / a.groupby("j").v.sum()).reindex(codes).fillna(0).values
    recs = []
    for y in range(Y0 + 1, Y1 + 1):
        recs.append(pd.DataFrame({
            "year": y, "iso3": ok, "g": g[y].values, "g_lag1": g[y - 1].values, "loggdp": gdp[y].values,
            "logpop": pop[y].values, "dpop": pop[y].values - pop[y - 1].values, "infl": infl[y].values, "emp": emp[y].values,
            "gmean": np.full(len(ok), g[y].mean()), "partner_ex_g": pex[y], "partner_im_g": pim[y],
            "exp_g": (lex[y] - lex[y - 1]).values, "imp_g": (lim[y] - lim[y - 1]).values,
            "fuel_share": fuel_sh[y].values, "hhi": hhi[y].values, "infl_imputed": infl[y].isna().values.astype(int) * 0}))
    P = pd.concat(recs); P.to_csv(OUT / "node_panel.csv.gz", index=False)
    rep.append(f"- Node panel: {len(P)} country-years, {P.shape[1]-2} features, years {Y0+1}-{Y1}")
    rep.append(f"- Extreme trade-growth values clipped later at the 1st/99th percentile of the training period")

    for y in range(Y0, Y1 + 1):
        t = pair[y]; a = t.groupby(["i", "j"]).v.sum().rename("v").reset_index()
        a = a.sort_values(["i", "v"], ascending=[True, False]).groupby("i").head(TOPK)
        a["xs"] = a.v / a.i.map(EX[y].set_axis(codes)).clip(lower=1)
        a["ms"] = a.v / a.j.map(IM[y].set_axis(codes)).clip(lower=1)
        gs = t.merge(a[["i", "j"]], on=["i", "j"]).pivot_table(index=["i", "j"], columns="g", values="v", fill_value=0)
        gs = gs.div(gs.sum(1), axis=0).reindex(columns=range(10), fill_value=0).add_prefix("gs").reset_index()
        a = a.merge(gs, on=["i", "j"], how="left").fillna(0); a["lv"] = np.log1p(a.v)
        a.to_csv(OUT / f"edges_{y}.csv.gz", index=False, compression="gzip")
    rep.append(f"- Edges: per year, each exporter keeps its top-{TOPK} destinations by total value (BACI, all products)")
    (OUT / "cleaning_report.md").write_text("# Cleaning report\n\n" + "\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
