# Sensitivity to the number of destinations kept per exporter (STGNN, base features, h16, wd 1e-3, 5 seeds)

The complete BACI network among the 182 countries has directed density 0.79 (mean 2016-2023). The model graph keeps each exporter's k largest destinations.

```
  k  edges_per_year  density  export_coverage  val_real  val_shuffled  test_real  test_shuffled    diff    DM_p
  5             910  0.02762          0.60449   0.01972       0.01972    0.03523        0.03521 0.00002 0.29177
 12            2184  0.06630          0.81149   0.01972       0.01973    0.03521        0.03521 0.00001 0.46654
 25            4550  0.13812          0.93648   0.01972       0.01972    0.03523        0.03521 0.00002 0.50297
all           25967  0.78828          1.00000   0.01972       0.01973    0.03521        0.03520 0.00001 0.63344
```

export_coverage = median share of an exporter's total exports captured by its top-k destinations; diff = mean (real - shuffled) test MAE; DM_p = year-clustered DM test (HLN, 7 years).