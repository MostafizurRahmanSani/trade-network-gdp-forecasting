# Supplementary runs (from saved runs)

## A. Graph controls on models that use the graph

Per-seed means over 5 seeds; ens = MAE of the seed-averaged forecast; diff = real minus control on seed-averaged forecasts (negative: real graph better); DM: year-clustered, HLN, t(6); paired t over seeds on per-seed test MAE.

| model | graph | val MAE | test MAE | seed sd | ens MAE | diff vs real | DM p | years real better | paired-seed p |
|---|---|---|---|---|---|---|---|---|---|
| STGNN (wd 0) | real | 0.02054 | 0.03702 | 0.00066 | 0.03597 | -- | -- | -- | -- |
| STGNN (wd 0) | shuffled | 0.02057 | 0.03680 | 0.00049 | 0.03592 | +0.00005 | 0.346 | 3 | 0.336 |
| STGNN (wd 0) | randedges | 0.02043 | 0.03685 | 0.00072 | 0.03589 | +0.00008 | 0.701 | 5 | 0.408 |
| STGNN (wd 0) | reproducibility: max abs difference between P1 real runs and the original runs STGNN_base_h16_wd0: 0.00e+00 | | | | | | | | |
| STGCN (wd 1e-3) | real | 0.02030 | 0.03577 | 0.00027 | 0.03574 | -- | -- | -- | -- |
| STGCN (wd 1e-3) | shuffled | 0.02021 | 0.03574 | 0.00026 | 0.03572 | +0.00002 | 0.815 | 3 | 0.043 |
| STGCN (wd 1e-3) | randedges | 0.02010 | 0.03601 | 0.00029 | 0.03587 | -0.00013 | 0.619 | 3 | 0.319 |
| STGCN (wd 1e-3) | reproducibility: max abs difference between P1 real runs and the original runs STGCN_base_h16_wd1e-3: 0.00e+00 | | | | | | | | |

### Shock propagation (2019 growth input of the shocked economy set to -0.05; change in the 2020 forecast; mean over 5 seeds; mean absolute change for countries linked to it in the real 2019 graph)

```
                                     own  receivers  own_exporters  non_neighbours  max_other  receivers/non-neighbours
model        graph     shocked                                                                                         
STGCN wd1e-3 randedges CHN     -2.22e-02   4.98e-05       1.50e-05        2.23e-05   3.68e-04                  2.24e+00
                       DEU     -7.38e-03   9.61e-07       7.70e-06        5.28e-06   1.39e-04                  1.82e-01
                       USA     -1.19e-02   4.15e-06       1.04e-05        1.10e-05   2.08e-04                  3.77e-01
             real      CHN     -1.82e-02   1.52e-05       1.31e-06        2.91e-10   2.47e-05                  5.22e+04
                       DEU     -4.13e-03   5.23e-06       5.69e-07        2.34e-10   9.86e-06                  2.23e+04
                       USA     -8.90e-03   7.47e-06       6.00e-07        0.00e+00   1.30e-05                       inf
             shuffled  CHN     -1.90e-02   1.84e-06       4.85e-07        2.81e-07   1.71e-05                  6.57e+00
                       DEU     -4.24e-03   1.61e-06       3.49e-07        4.07e-07   1.37e-05                  3.95e+00
                       USA     -9.41e-03   8.29e-07       4.83e-07        2.18e-07   1.44e-05                  3.81e+00
STGNN wd0    randedges CHN     -2.88e-02   2.21e-05       1.06e-05        1.60e-05   4.27e-04                  1.38e+00
                       DEU     -7.75e-03   8.49e-06       8.82e-06        4.74e-06   2.80e-04                  1.79e+00
                       USA     -1.36e-02   1.05e-05       9.56e-06        1.13e-05   3.58e-04                  9.36e-01
             real      CHN     -3.01e-02   6.33e-05       5.43e-06        0.00e+00   1.59e-04                       inf
                       DEU     -9.19e-03   3.53e-05       3.82e-06        0.00e+00   1.17e-04                       inf
                       USA     -1.46e-02   4.36e-05       3.49e-06        0.00e+00   1.18e-04                       inf
             shuffled  CHN     -2.96e-02   7.78e-06       7.02e-06        1.02e-05   5.39e-04                  7.64e-01
                       DEU     -7.81e-03   5.98e-06       3.65e-06        5.66e-06   3.48e-04                  1.06e+00
                       USA     -1.42e-02   1.13e-05       5.13e-06        3.69e-07   2.73e-04                  3.06e+01
```

Actual 2019 growth of the shocked economies (the shock replaces it with -0.05): CHN +0.0589, USA +0.0255, DEU +0.0097

Selected STGNN (wd 1e-3, propagation.md) for comparison:

```
              own  receivers  non_neighbours
shocked                                     
CHN     -2.13e-02   0.00e+00        0.00e+00
DEU     -1.16e-03   0.00e+00        0.00e+00
USA     -4.92e-03   0.00e+00        0.00e+00
```

## B. CP-protocol configurations selected on the early-stopping targets 2013-2014 only

| model | hidden | wd | 5-seed MAE 2013-14 | calibration MAE 2015-17 | test MAE | selected (2013-14) | used before |
|---|---|---|---|---|---|---|---|
| STGNN | 16 | 0 | 0.02178 | 0.02155 | 0.03759 |  |  |
| STGNN | 16 | 1e-3 | 0.02011 | 0.02004 | 0.03537 | yes | yes |
| STGNN | 32 | 0 | 0.02083 | 0.02062 | 0.03638 |  |  |
| STGNN | 32 | 1e-3 | 0.02022 | 0.02004 | 0.03540 |  |  |
| GRU | 16 | 0 | 0.02141 | 0.02069 | 0.03722 |  |  |
| GRU | 16 | 1e-3 | 0.01986 | 0.01978 | 0.03653 | yes | yes |
| GRU | 32 | 0 | 0.02118 | 0.02043 | 0.03694 |  |  |
| GRU | 32 | 1e-3 | 0.01987 | 0.01999 | 0.03654 |  |  |
| LSTM | 16 | 0 | 0.02119 | 0.02040 | 0.03708 |  |  |
| LSTM | 16 | 1e-3 | 0.01981 | 0.02003 | 0.03662 | yes | yes |
| LSTM | 32 | 0 | 0.02077 | 0.02062 | 0.03734 |  |  |
| LSTM | 32 | 1e-3 | 0.01982 | 0.01999 | 0.03641 |  |  |

## C. GRU and LSTM under the STGNN training protocol (full batch per training year, chronological, no clipping)

| model | protocol | val MAE | test MAE | seed sd | ens MAE |
|---|---|---|---|---|---|
| GRU (h16) | mini-batch 32, clipping | 0.01943 | 0.03646 | 0.00035 | 0.03625 |
| GRU (h16) | STGNN protocol | 0.02003 | 0.03655 | 0.00016 | 0.03652 |
| LSTM (h16) | mini-batch 32, clipping | 0.01950 | 0.03634 | 0.00006 | 0.03623 |
| LSTM (h16) | STGNN protocol | 0.01986 | 0.03650 | 0.00007 | 0.03647 |
| STGNN | STGNN protocol | 0.01972 | 0.03521 | 0.00007 | 0.03517 |
| STGNN, 0 message-passing layers | STGNN protocol | 0.01977 | 0.03524 | 0.00005 | 0.03515 |

| comparison (seed-averaged forecasts) | diff | DM p | years first better |
|---|---|---|---|
| P3fb_GRU_base_h16_wd1e-3 vs GRU_base_h16_wd1e-3 | +0.00027 | 0.259 | 2 |
| P3fb_LSTM_base_h16_wd1e-3 vs LSTM_base_h16_wd1e-3 | +0.00023 | 0.108 | 1 |
| STGNN_base_h16_wd1e-3 vs P3fb_GRU_base_h16_wd1e-3 | -0.00135 | 0.338 | 3 |
| STGNN_base_h16_wd1e-3 vs P3fb_LSTM_base_h16_wd1e-3 | -0.00130 | 0.325 | 4 |
| STGNNX-l0_base_h16_wd1e-3 vs P3fb_GRU_base_h16_wd1e-3 | -0.00136 | 0.325 | 5 |

## D. Gate activations of the message-passing GRU cells (final STGNN, retrained with the train_neural procedure; test windows, all 6 years x 182 economies x 7 samples)

Test MAE of the retrained models 0.03521; max abs difference to the original runs 0.00e+00.

Mean over seeds (sd over seeds in the second table). update = z (PyTorch GRUCell: new state = (1 - z) * candidate + z * old state); reset = r; message/state term = mean |W x| of the input (aggregated message) and hidden (own state) weight products, biases excluded.

```
layer                    layer1
update_mean              0.4999
update_sd              0.001637
update_min               0.4951
update_max               0.5027
reset_mean                  0.5
reset_sd              0.0001055
mean_abs_message_term 1.737e-07
mean_abs_state_term   0.0004043
mean_abs_bias_ih       0.007196
mean_abs_bias_hh       0.005014
norm_W_ih              0.003417
norm_W_hh               0.05018
n                          7644
```

```
layer                  layer1
update_mean           0.00078
update_sd              0.0028
update_min               0.01
update_max             0.0028
reset_mean            3.1e-05
reset_sd              8.3e-05
mean_abs_message_term 3.3e-07
mean_abs_state_term   0.00068
mean_abs_bias_ih       0.0043
mean_abs_bias_hh       0.0043
norm_W_ih              0.0045
norm_W_hh               0.065
n                           0
```
