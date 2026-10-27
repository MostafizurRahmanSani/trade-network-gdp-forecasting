# Outputs

| Path | Content |
|---|---|
| `runs/` | one `.json` (validation and test MAE, run time) and one `.npz` (validation, test and, for conformal-protocol runs, calibration predictions) per neural run; naming scheme in `scripts/README.md` |
| `baselines/` | predictions of the non-neural models (`<model>.npz` with `val_pred`, `val_true`, `test_pred`, `test_true`) |
| `stability/` | training curves and checkpoint-rule results of the STGNN (five seeds) |
| `point_accuracy.csv`, `posthoc_tests.md` | accuracy of all models with Diebold-Mariano, Holm, sign-flip and model-confidence-set tests (Table 5 of the paper) |
| `final_tables.md`, `final_accuracy.csv` | accuracy tables, per-year and per-group errors, graph controls |
| `imf_comparison.md`, `arima_drift_report.md` | comparison with the IMF forecasts |
| `ablation_final.md`, `topk_sensitivity.md`, `propagation.md`, `graph_analysis.md`, `graph_statistics.csv` | ablation, graph density, shock propagation, learned gates and graph statistics |
| `rolling_origins.md/.csv`, `horizon2.md`, `sample_sensitivity.md`, `robustness_report.md`, `additional_tables.md`, `supplementary_report.md` | robustness checks |
| `conformal_results.csv/.md`, `conformal_posthoc.md`, `conformal_extras.md`, `conformal_ranges.md` | conformal prediction results |
| `tuning_extension.md`, `tuning_decisions.json` | extended tuning check and its decisions (none adopted) |

All files are produced by the scripts in `scripts/`; see `scripts/README.md`.
