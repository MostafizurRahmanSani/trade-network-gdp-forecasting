# Scripts

Run everything from the repository root (`python scripts/<name>.py`). Scripts skip work that already has a result file, so an interrupted stage can simply be restarted. Training uses one CPU thread per run; use at most 3-4 parallel workers.

## Libraries (not run directly)

| File | Purpose |
|---|---|
| `panel_data.py` | builds the windows of yearly country graphs and the train / validation / test split |
| `train_neural.py` | model factory and `run_one` (one run = model, feature set, hidden size, weight decay, seed); `grid` mode trains the four selection configurations with five seeds |
| `selection.py` | validation-based selection of the final configuration of each model |
| `stgnn_variants.py` | configurable STGNN (`STGNNX-l1-evec-d2` is the selected model), Bönnig-style GAT, MTGNN-style, TFT-style and Transformer baselines |
| `tabwin.py`, `xlsx_reader.py` | helpers for the tabular baselines and the IMF spreadsheet |

## Stages

**0. Data** (needs the raw downloads, see `data/README.md`; the processed files are already included)
`01_aggregate_baci.py` -> `02_build_panel.py` -> `03_edges_topk.py` -> `04_build_panel_extended.py`

**1. Non-neural baselines** (`outputs/baselines/*.npz`)
`10_baselines_pooled.py`, `11_baselines_panel_gvar.py`, `12_arima_rolling.py`, `13_arima_drift.py` (and `13_arima_drift.py cp`), `14_arima_cp_protocol.py`, `15_dynamic_factor.py`, `16_gvar_country.py`, `17_tabular_base_features.py`, `18_tabular_window.py`, `19_random_forest_seeds.py`, `20_tabular_eleven_features.py`

**2. Neural models** (`outputs/runs/*`), in this order
1. `python scripts/train_neural.py grid --models <all families> --procs 3` - the four configurations per family, five seeds (selection by validation MAE)
2. `31_train_extended_tuning.py`, then `57_tuning_report.py` (writes `outputs/tuning_decisions.json`), then `32_adopt_extended_tuning.py` (trains adopted configurations; none are adopted in the reported results)
3. `33_train_graph_controls.py`, `39_train_protocol_variants.py`, `34_train_topk.py`, `35_train_extended_sample.py`, `40_shock_propagation.py`, `36_train_publication_lag.py`, `37_train_rolling_origins.py`, `38_train_ablation.py`, `41_training_stability.py`
4. `42_train_supplementary.py P1 P2 P3 P9`, `43_refit_to_2017.py`, `44_conformal_prediction.py`

**3. Reports, tables and figures** (from the saved runs, no training except `55_graph_analysis.py`, which trains one STGNN)
`bash scripts/run_reports.sh` runs `50_...` to `72_...` (tables and reports into `outputs/`) and `80_...` to `83_...` (figures into `figures/`).

## Run-name prefixes in `outputs/runs`

A run file is named `<model>[@<variant>][#lr<value>]_<features>_h<hidden>_wd<weight decay>_s<seed>.{json,npz}`.

- `@gfc`, `@mid`: rolling-origin retraining (test targets 2009-10 and 2013-16); `@cp`: conformal-protocol split; `@h2`: publication-lag scenario; `@ext`: extended sample; `@k5`, `@k25`, `@kall`: top-k graphs; `@pynorm`: per-year normalisation
- `#lr0.0003`, `#lr0.003`: extended tuning check (learning rate)
- `STGNNX-...`: ablation variants of the STGNN (`stgnn_variants.py`)
- `STGNN_shuffled`, `_uniform`, `_shufw`, `_randedges`, `_static`, `_geo`, `_reversed`, `_undirected`: graph controls
- `P1_...`: graph controls of the STGNN without weight decay and of STGCN; `P3fb_...`: GRU and LSTM trained with the STGNN's full-batch procedure; `P9_...`: STGNN retrained with the gate activations recorded; `REFIT17_...`: refit on targets up to 2017
