#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs figures
for s in 50_collect_results 51_accuracy_tables 52_rolling_origins_report 53_additional_tables 54_per_country 55_graph_analysis 56_propagation_report \
         57_tuning_report 58_ablation_report 59_topk_report 60_sample_sensitivity_report 61_publication_lag_report 62_imputation_report \
         63_imputation_lookahead 64_shock_cases 65_imf_comparison 66_posthoc_tests 67_arima_drift_report 68_conformal_posthoc 69_conformal_extras \
         70_supplementary_report 71_conformal_ranges 72_robustness_report 80_figure_training_curves 81_figures_summary 82_figures_models_and_controls \
         83_figure_conformal_coverage; do
  echo "=== $s $(date +%H:%M:%S)"
  python -I scripts/$s.py > logs/$s.log 2>&1 && echo "OK $s" || { echo "FAIL $s"; tail -n 3 logs/$s.log; }
done
echo "REPORTS-DONE"
