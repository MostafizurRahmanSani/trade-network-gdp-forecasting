# Data

## Sources

| Data | Source | Included here |
|---|---|---|
| Bilateral trade (edges) | CEPII BACI, HS92 classification, release V202601 (22 January 2026), exporter x importer x HS6 flows. Gaulier and Zignago (2010), http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37 | the derived edge files in `processed/` (top-12 export destinations per exporter, 13 aggregated attributes); the raw yearly files (8 GB, `raw/baci/`) and the product-group aggregates (`interim/`) exist only locally and are git-ignored |
| Node variables and target | World Bank, World Development Indicators (NY.GDP.MKTP.KD, SP.POP.TOTL, FP.CPI.TOTL.ZG, SL.EMP.TOTL.SP.ZS), extract downloaded on 29 January 2026 from https://databank.worldbank.org/source/world-development-indicators (series codes as listed) | `raw/wdi/` (WDI_extract.csv, series metadata and the original download; 2 MB) |
| IMF forecasts | IMF World Economic Outlook Historical Forecasts Database (`WEOhistorical.xlsx`, file updated 18 October 2022), https://www.imf.org/external/pubs/ft/weo/data/WEOhistorical.xlsx | `external/weo/weohistorical.xlsx` (needed by `scripts/65_imf_comparison.py` and `67_arima_drift_report.py`) |
| Country codes and centroids | CEPII `country_codes_V202601.csv`; country centroids (Hong Kong and Macao added by hand) | `external/` |

Licences: CEPII BACI is released under the Etalab Open Licence 2.0 (as stated on the CEPII BACI page, checked October 2026), which authorises reuse and redistribution provided the source is mentioned; cite Gaulier and Zignago (2010), BACI: International Trade Database at the Product-Level. The 1994-2007 Version, CEPII Working Paper 2010-23. BACI is built from UN Comtrade data, whose own terms restrict the redistribution of raw Comtrade data without permission of the United Nations; the raw BACI files are therefore not part of the GitHub repository (they are git-ignored), and the repository contains only aggregated and derived files (trade summed over product groups, top-12 destinations per exporter). The World Bank data are published under CC BY 4.0 (Source: World Bank, World Development Indicators, https://datacatalog.worldbank.org/search/dataset/0037712); the extract in `raw/wdi/` is redistributed unchanged with this credit. The files derived from BACI are modified: trade flows are aggregated over product groups, restricted to each exporter's top-12 destinations and combined with World Bank data, so they are not the original BACI files. The IMF World Economic Outlook file is published statistical data. The IMF's terms for such data (https://www.imf.org/en/about/copyright-and-terms, checked October 2026) allow reproduction and adaptation provided the IMF is credited and material transformations are stated. Source: International Monetary Fund, World Economic Outlook database (historical forecasts). The file is included unchanged. Commercial reuse needs permission from copyright@imf.org.

## Layout

```
processed/        countries.csv, node_panel.csv.gz, edges_<year>.csv.gz (top-12), edges_k<k>_<year>.csv.gz (k = 5, 25, all), cleaning_report.md
processed_ext/    the same files for the extended sample (189 economies), used by the sample-selection check
interim/          (git-ignored) BACI aggregated to exporter x importer x product group (pair_group_<year>.csv.gz), written by 01_aggregate_baci.py, the input of 02_build_panel.py
external/         country_codes_V202601.csv, countries_centroids.csv, weo/weohistorical.xlsx (IMF forecasts)
raw/wdi/          World Bank extract, series metadata and the original download
raw/baci/         the raw BACI yearly files BACI_HS92_Y<year>_V202601.csv, 8 GB (git-ignored); needed only for step 1 below
```

## Rebuilding the processed files

1. (Optional) download the raw BACI files from CEPII into `raw/baci/` and run `python scripts/01_aggregate_baci.py`; it writes `interim/`.
2. `python scripts/02_build_panel.py` builds `processed/` from `interim/` and the World Bank extract (rebuilt locally, the files are identical in content to the ones included).

All models and reports run from `processed/` (and `processed_ext/`, `external/`) alone; `raw/baci/` and `interim/` are needed only for step 1, and `raw/wdi/` (included) for step 2 and by `62_imputation_report.py` and `63_imputation_lookahead.py`).
3. `scripts/03_edges_topk.py` and `scripts/04_build_panel_extended.py` build the top-k and extended-sample files.

## Panel

182 economies present in BACI and WDI with complete real GDP and population for 1994-2024; yearly directed graphs 1995-2024 in which every exporter keeps its 12 largest destinations (2,093-2,184 edges per year); 6 node features (growth, log real GDP, log population, population growth, inflation, employment-to-population ratio) and 13 edge features (log trade value, exporter share, importer share, shares of ten product groups). Samples are indexed by the last input year t (window t-5..t, target = growth of year t+1): training t = 2001-2013, validation 2014-2016, test 2017-2023.
