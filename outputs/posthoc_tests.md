# Post-hoc analyses on saved predictions (no model refitted)

## A. Table 5 extended

test_MAE = mean of per-seed MAE; ens_MAE = MAE of the seed-averaged forecast (used by all DM tests); rel_rwdrift = MAE / MAE of the random walk with drift; dm_p = year-clustered DM (HLN, t(6)) of the STGNN against the model, on seed-averaged forecasts; W = years (of 7) the STGNN is better; p_rand = exact sign-flip randomisation test of the mean yearly differential (128 sign patterns); p_sign = two-sided sign test on W; dm_p_holm = Holm adjustment over the 33 comparisons of the STGNN with the other models of Table 5; mcs_p = model-confidence-set p-value (T_max, i.i.d. bootstrap over the 7 test years, 10,000 draws) over the 34 models of Table 5 (variants excluded); a model is in the 90% MCS if mcs_p >= 0.10.
Info: N6 = six base node features; +G = trade graph; tabular models of Table 5 receive the N6 window (36 inputs); Y = own growth history; +W = cross-country mean growth; +T = trade-weighted partner growth.

```
                                                                  model       info  seeds  val_MAE  test_MAE       sd  ens_MAE    RMSE   MASE  rel_nochange  rel_rwdrift  exCOVID   COVID  dm_diff   dm_p    W  p_rand  p_sign  dm_p_holm  mcs_p
key                                                                                                                                                                                                                                             
STGNNX-d2                                    STGNN variant: dropout 0.2       N6+G      5  0.01969   0.03511  0.00015  0.03506  0.0670  0.803         0.729        0.908   0.0222  0.0675  0.00011  0.290  1.0   0.203   0.125        NaN    NaN
STGNNX               STGNN ablation reference (per-layer edge networks)       N6+G      5  0.01975   0.03512  0.00013  0.03506  0.0669  0.802         0.729        0.908   0.0222  0.0675  0.00011  0.586  3.0   0.641   1.000        NaN    NaN
STGNN                                                             STGNN       N6+G      5  0.01972   0.03521  0.00006  0.03517  0.0671  0.804         0.731        0.911   0.0222  0.0678      NaN    NaN  NaN     NaN     NaN        NaN  1.000
STGNNX-l0                  STGNN variant: no message passing (0 layers)         N6      5  0.01977   0.03524  0.00004  0.03515  0.0670  0.805         0.732        0.912   0.0222  0.0678  0.00001  0.897  5.0   0.859   0.453        NaN    NaN
RandomForest_win                                          Random forest         N6      5  0.02006   0.03549  0.00002  0.03549  0.0670  0.814         0.737        0.918   0.0226  0.0676 -0.00032  0.482  4.0   0.438   1.000      1.000  0.898
GWN                                                 Graph WaveNet-style       N6+G      5  0.02048   0.03567  0.00003  0.03563  0.0672  0.817         0.741        0.923   0.0230  0.0673 -0.00046  0.199  5.0   0.188   0.453      1.000  0.898
STGNNX-k1                   STGNN variant: skip from last-year features       N6+G      5  0.01925   0.03571  0.00021  0.03562  0.0674  0.819         0.742        0.924   0.0219  0.0701 -0.00045  0.655  2.0   0.938   0.453        NaN    NaN
STGCN                                                             STGCN       N6+G      5  0.02030   0.03577  0.00024  0.03574  0.0672  0.821         0.743        0.925   0.0228  0.0682 -0.00057  0.029  6.0   0.047   0.125      0.685  0.888
GBM_win                                               Gradient boosting         N6      1  0.01965   0.03580      NaN  0.03580  0.0677  0.818         0.743        0.926   0.0226  0.0688 -0.00063  0.013  7.0   0.016   0.016      0.329  0.888
GAT                                                                 GAT       N6+G      5  0.01943   0.03582  0.00018  0.03579  0.0673  0.822         0.744        0.926   0.0222  0.0698 -0.00062  0.447  3.0   0.562   1.000      1.000  0.888
MTGNN_trade                                    MTGNN-style, trade graph       N6+G      5  0.01939   0.03591  0.00032  0.03566  0.0674  0.820         0.746        0.929   0.0224  0.0696 -0.00049  0.508  4.0   0.703   1.000      1.000  0.898
STGNNX-tlast                        STGNN variant: last input year only       N6+G      5  0.01923   0.03597  0.00025  0.03581  0.0676  0.823         0.747        0.930   0.0222  0.0705 -0.00064  0.528  3.0   0.844   1.000        NaN    NaN
MTGNN                                                       MTGNN-style       N6+G      5  0.01937   0.03607  0.00029  0.03576  0.0677  0.824         0.749        0.933   0.0226  0.0698 -0.00059  0.353  6.0   0.391   0.125      1.000  0.888
TFT                                                           TFT-style         N6      5  0.01903   0.03608  0.00016  0.03593  0.0673  0.831         0.749        0.933   0.0221  0.0711 -0.00077  0.585  3.0   0.859   1.000      1.000  0.888
GNN                                                                 GNN       N6+G      5  0.01921   0.03609  0.00011  0.03594  0.0677  0.826         0.749        0.933   0.0223  0.0705 -0.00078  0.441  3.0   0.688   1.000      1.000  0.888
Transformer                                                 Transformer         N6      5  0.01934   0.03617  0.00026  0.03602  0.0676  0.829         0.751        0.936   0.0224  0.0706 -0.00085  0.442  3.0   0.688   1.000      1.000  0.888
BiLSTM                                                           BiLSTM         N6      5  0.01944   0.03628  0.00018  0.03609  0.0676  0.839         0.753        0.938   0.0226  0.0704 -0.00092  0.405  4.0   0.453   1.000      1.000  0.888
DCRNN                                                             DCRNN       N6+G      5  0.02002   0.03630  0.00012  0.03618  0.0677  0.836         0.754        0.939   0.0227  0.0702 -0.00102  0.169  5.0   0.109   0.453      1.000  0.886
LSTM                                                               LSTM         N6      5  0.01950   0.03634  0.00005  0.03623  0.0677  0.840         0.755        0.940   0.0224  0.0712 -0.00107  0.385  3.0   0.484   1.000      1.000  0.886
GRU                                                                 GRU         N6      5  0.01943   0.03646  0.00031  0.03625  0.0675  0.846         0.757        0.943   0.0225  0.0714 -0.00108  0.419  3.0   0.469   1.000      1.000  0.886
SVR_win                                                             SVR         N6      1  0.01973   0.03660      NaN  0.03660  0.0689  0.843         0.760        0.947   0.0219  0.0734 -0.00144  0.388  4.0   0.453   1.000      1.000  0.832
Ridge_win                                                         Ridge         N6      1  0.01963   0.03666      NaN  0.03666  0.0681  0.847         0.761        0.948   0.0220  0.0733 -0.00149  0.468  3.0   0.875   1.000      1.000  0.886
RNN                                                                 RNN         N6      5  0.01920   0.03686  0.00026  0.03660  0.0683  0.856         0.765        0.953   0.0220  0.0741 -0.00143  0.541  2.0   0.938   0.453      1.000  0.888
STGAT                                                             STGAT       N6+G      5  0.02015   0.03695  0.00034  0.03618  0.0682  0.845         0.767        0.956   0.0237  0.0700 -0.00102  0.250  4.0   0.250   1.000      1.000  0.886
MedianReg_win                                         Median regression         N6      1  0.01950   0.03714      NaN  0.03714  0.0689  0.860         0.771        0.961   0.0221  0.0747 -0.00197  0.383  4.0   0.438   1.000      1.000  0.794
AR1+globalmean                               AR(1) pooled + global mean        Y+W      1  0.02121   0.03795      NaN  0.03795  0.0695  0.887         0.788        0.982   0.0241  0.0725 -0.00278  0.018  6.0   0.031   0.125      0.461  0.485
AR1_pooled                                                 AR(1) pooled          Y      1  0.02119   0.03811      NaN  0.03811  0.0697  0.891         0.791        0.986   0.0241  0.0730 -0.00294  0.030  6.0   0.031   0.125      0.685  0.291
GCN                                                                 GCN       N6+G      5  0.02301   0.03812  0.00051  0.03780  0.0693  0.877         0.792        0.986   0.0260  0.0685 -0.00263  0.057  6.0   0.047   0.125      1.000  0.793
BonnigGAT                                                    Bonnig GAT       N6+G      5  0.02228   0.03813  0.00028  0.03793  0.0692  0.883         0.792        0.986   0.0254  0.0700 -0.00276  0.000  7.0   0.016   0.016      0.010  0.540
BonnigGAT_edges                               Bonnig GAT, edge features       N6+G      5  0.02196   0.03864  0.00014  0.03851  0.0696  0.882         0.802        0.999   0.0259  0.0705 -0.00334  0.001  7.0   0.016   0.016      0.044  0.291
RW_drift                                         Random walk with drift          Y      1  0.02469   0.03866      NaN  0.03866  0.0701  0.857         0.803        1.000   0.0263  0.0696 -0.00350  0.006  6.0   0.031   0.125      0.159  0.540
GVAR_style                                            GVAR-type, pooled        Y+T      1  0.02059   0.03900      NaN  0.03900  0.0697  0.916         0.810        1.009   0.0242  0.0760 -0.00383  0.122  7.0   0.016   0.016      1.000  0.231
ARIMA_rolling_drift                            Rolling ARIMA with drift          Y      1  0.02189   0.03900      NaN  0.03900  0.0720  0.893         0.810        1.009   0.0238  0.0769 -0.00383  0.218  5.0   0.062   0.453      1.000  0.485
PanelAR2_FE                                  Panel AR(2), fixed effects          Y      1  0.02356   0.03961      NaN  0.03961  0.0704  0.886         0.823        1.024   0.0268  0.0717 -0.00444  0.000  7.0   0.016   0.016      0.002  0.225
GVAR_country                                GVAR-type, country-specific        Y+T      1  0.02332   0.04023      NaN  0.04023  0.0707  0.911         0.835        1.041   0.0263  0.0750 -0.00506  0.003  7.0   0.016   0.016      0.084  0.000
DynamicFactor                                      Dynamic factor model  Y (panel)      1  0.02341   0.04060      NaN  0.04060  0.0711  0.898         0.843        1.050   0.0281  0.0717 -0.00543  0.001  7.0   0.016   0.016      0.025  0.019
CountryMeanGrowth                                   Country mean growth          Y      1  0.02529   0.04075      NaN  0.04075  0.0725  0.887         0.846        1.054   0.0288  0.0706 -0.00558  0.004  6.0   0.031   0.125      0.109  0.255
Persistence(level)                                            No-change          -      1  0.03878   0.04816      NaN  0.04816  0.0718  1.090         1.000        1.245   0.0414  0.0650 -0.01299  0.100  6.0   0.125   0.125      1.000  0.194
LastGrowth                      Growth persistence (last year's growth)          Y      1  0.02204   0.04884      NaN  0.04884  0.0913  1.128         1.014        1.263   0.0278  0.1014 -0.01367  0.273  5.0   0.156   0.453      1.000  0.540
```

90% MCS contains 32 of 34 models; excluded: GVAR-type, country-specific, Dynamic factor model.

Ranking of the 34 models by validation MAE (top 10): TFT-style 0.01903, RNN 0.01920, GNN 0.01921, Transformer 0.01934, MTGNN-style 0.01937, MTGNN-style, trade graph 0.01939, GAT 0.01943, GRU 0.01943, BiLSTM 0.01944, LSTM 0.01950

Rank of the STGNN by validation MAE among the 34 models: 14; by test MAE: 1.

Test-MAE rank of 'last input year only' if it replaced the STGNN among the 34 models: 7 of 34.

## B. Bias, sign of forecasts and shrinkage (seed-averaged forecasts)

Negative outcomes in the test period: 255 of 1274 country-years (2018: 16, 2019: 24, 2020: 147, 2021: 12, 2022: 17, 2023: 21, 2024: 18).

| Model | min forecast | 5th pct | 95th pct | negative forecasts | hit rate on negative outcomes | sign accuracy | bias 2018 | bias 2019 | bias 2020 | bias 2021 | bias 2022 | bias 2023 | bias 2024 | bias all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STGNN | 0.0109 | 0.0150 | 0.0579 | 0 | 0.000 | 0.799 | +0.0027 | +0.0070 | +0.0869 | -0.0307 | -0.0071 | +0.0057 | +0.0048 | +0.0099 |
| STGNN variant: no message passing (0 layers) | 0.0098 | 0.0145 | 0.0585 | 0 | 0.000 | 0.799 | +0.0027 | +0.0071 | +0.0869 | -0.0311 | -0.0076 | +0.0050 | +0.0042 | +0.0096 |
| STGNN variant: last input year only | 0.0007 | 0.0065 | 0.0661 | 0 | 0.000 | 0.799 | +0.0033 | +0.0069 | +0.0865 | -0.0393 | +0.0017 | +0.0042 | +0.0008 | +0.0092 |
| Random forest | 0.0086 | 0.0135 | 0.0611 | 0 | 0.000 | 0.799 | +0.0030 | +0.0069 | +0.0857 | -0.0332 | -0.0006 | +0.0079 | +0.0035 | +0.0105 |
| SVR | -0.0296 | -0.0006 | 0.0641 | 68 | 0.035 | 0.760 | +0.0044 | +0.0088 | +0.0885 | -0.0435 | -0.0041 | +0.0011 | +0.0039 | +0.0084 |
| Gradient boosting | -0.0216 | 0.0074 | 0.0646 | 29 | 0.043 | 0.794 | +0.0057 | +0.0089 | +0.0877 | -0.0334 | +0.0012 | +0.0022 | +0.0027 | +0.0107 |
| GRU | -0.0365 | 0.0011 | 0.0568 | 59 | 0.059 | 0.776 | +0.0024 | +0.0070 | +0.0868 | -0.0419 | -0.0106 | +0.0037 | +0.0035 | +0.0073 |
| LSTM | -0.0172 | 0.0034 | 0.0556 | 44 | 0.047 | 0.783 | +0.0026 | +0.0071 | +0.0870 | -0.0407 | -0.0107 | +0.0035 | +0.0035 | +0.0075 |
| AR(1) pooled | -0.1808 | 0.0085 | 0.0526 | 37 | 0.039 | 0.786 | +0.0062 | +0.0103 | +0.0899 | -0.0424 | +0.0013 | +0.0090 | +0.0074 | +0.0117 |
| Random walk with drift | -0.0362 | 0.0065 | 0.0708 | 17 | 0.016 | 0.792 | +0.0062 | +0.0106 | +0.0911 | -0.0233 | -0.0083 | +0.0036 | +0.0049 | +0.0121 |

Bias = mean of (forecast - outcome). Hit rate = share of negative outcomes with a negative forecast. Sign accuracy = share of country-years with the correct sign.

- VEN: outcomes -0.219, -0.324, -0.357, +0.010, +0.077, +0.039, +0.052; STGNN +0.021, +0.020, +0.020, +0.016, +0.018, +0.042, +0.040
- LBY: outcomes +0.076, -0.119, -0.349, +0.249, -0.087, +0.097, +0.019; STGNN +0.045, +0.050, +0.033, +0.025, +0.050, +0.032, +0.050
- SDN: outcomes -0.027, -0.022, -0.037, -0.019, -0.010, -0.349, -0.150; STGNN +0.031, +0.031, +0.030, +0.029, +0.031, +0.033, +0.029

## C. Graph controls, paired by seed (real minus control; per-seed test MAE differences, mean and sd over the 5 seeds; paired t-test) and on seed-averaged forecasts

| Control | per-seed diff mean | sd | paired t p | seed-averaged diff | DM p |
|---|---|---|---|---|---|
| shuffled | +0.00001 | 0.00000 | 0.001 | +0.00001 | 0.347 |
| uniform | -0.00006 | 0.00017 | 0.460 | -0.00003 | 0.843 |
| shufw | +0.00001 | 0.00001 | 0.168 | +0.00001 | 0.262 |
| randedges | +0.00002 | 0.00001 | 0.035 | +0.00003 | 0.619 |
| static | +0.00001 | 0.00001 | 0.084 | +0.00001 | 0.023 |
| geo | +0.00001 | 0.00001 | 0.216 | +0.00002 | 0.643 |
| undirected | -0.00002 | 0.00009 | 0.672 | +0.00001 | 0.872 |
| reversed | -0.00002 | 0.00008 | 0.609 | +0.00000 | 0.965 |

The year-clustered DM statistic divides a mean differential by its standard deviation; for two almost identical forecasts both are tiny, so differences of a few 1e-5 can be 'significant'.

## D. Top-k sensitivity recomputed on seed-averaged forecasts (same runs as topk_sensitivity.md, which averaged per-seed absolute errors)

| k | diff (real - shuffled) | DM p | per-seed-error convention diff (topk_sensitivity.md) |
|---|---|---|---|
| 5 | +0.00002 | 0.382 | +0.00002 |
| 12 | +0.00001 | 0.347 | +0.00001 |
| 25 | +0.00001 | 0.618 | +0.00002 |
| all | +0.00001 | 0.645 | +0.00001 |

Edges per year in topk_sensitivity.md are means over 2016-2023 (k=12: 2184; range over all years in graph_analysis.md).

## E. STGNN without weight decay

Validation MAE 0.02054 (seed sd 0.00059); test MAE 0.03702 (seed sd 0.00066, population sd 0.00059).

## F. Robustness variants of the ML regressors (not in Table 5): 11 node features (no partner growth) versus 13 features (with trade-weighted partner growth), each with four growth lags

| Model | val 11 | val 13 | test 11 | test 13 | DM p (11 vs 13) | years 13 better |
|---|---|---|---|---|---|---|
| Ridge | 0.02018 | 0.02075 | 0.03650 | 0.03717 | 0.179 | 2 |
| MedianReg | 0.01963 | 0.01967 | 0.03711 | 0.03717 | 0.239 | 3 |
| GBM | 0.01967 | 0.01971 | 0.03579 | 0.03575 | 0.593 | 2 |

Random forest and SVR were run only with the 13-feature set (11_baselines_panel_gvar.py).

## G. Test MAE without the samples whose input window contains a future-dependent interpolated value

Samples removed: 6 of 1274.

| Model | all | without affected samples |
|---|---|---|
| STGNN | 0.03517 | 0.03499 |
| Random forest | 0.03549 | 0.03529 |
| SVR | 0.03660 | 0.03643 |
| Gradient boosting | 0.03580 | 0.03559 |
| GAT | 0.03579 | 0.03559 |
| STGCN | 0.03574 | 0.03556 |
| LSTM | 0.03623 | 0.03606 |
| GRU | 0.03625 | 0.03607 |
| AR(1) pooled | 0.03811 | 0.03791 |
| Random walk with drift | 0.03866 | 0.03849 |
| No-change | 0.04816 | 0.04807 |

## H. IMF October forecast: year-clustered DM tests on the country-years with an IMF forecast (6 years, HLN, t(5))

Country-years: 1061. Economies without an IMF October forecast in at least one year: ABW, AND, BMU, CUB, ETH, PSE, PYF, UKR.

| Comparison | mean loss diff | DM p | years first better (of 6) |
|---|---|---|---|
| STGNN vs IMF_fall | +0.00304 | 0.086 | 1 |
| Random forest vs IMF_fall | +0.00347 | 0.097 | 1 |
| Random walk with drift vs IMF_fall | +0.00633 | 0.000 | 0 |
| STGNN vs RandomForest_win | -0.00043 | 0.446 | 4 |
