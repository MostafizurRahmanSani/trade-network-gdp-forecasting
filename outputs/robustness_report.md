# Robustness checks (round 3)

## 1. Tabular regressors with three input sets (single fit, random_state 0 where random)

| model | input set | file | val MAE | test MAE | DM p vs STGNN (STGNN better in years) |
|---|---|---|---|---|---|
| Ridge | 6 features + 4 lags | Ridge_base | 0.01971 | 0.03654 | 0.457 (3) |
| Ridge | 11 features + 4 lags | Ridge_ext | 0.02018 | 0.03650 | 0.389 (5) |
| Ridge | 13 features + 4 lags | Ridge_partner | 0.02075 | 0.03717 | 0.125 (6) |
| Median regression | 6 features + 4 lags | MedianReg_base | 0.01956 | 0.03709 | 0.400 (5) |
| Median regression | 11 features + 4 lags | MedianReg_ext | 0.01963 | 0.03711 | 0.368 (5) |
| Median regression | 13 features + 4 lags | MedianReg_partner | 0.01967 | 0.03717 | 0.363 (5) |
| Gradient boosting | 6 features + 4 lags | GBM_base | 0.01976 | 0.03585 | 0.140 (5) |
| Gradient boosting | 11 features + 4 lags | GBM_ext | 0.01967 | 0.03579 | 0.300 (5) |
| Gradient boosting | 13 features + 4 lags | GBM_partner | 0.01971 | 0.03575 | 0.367 (4) |
| Random forest | 6 features + 4 lags | RandomForest_base | 0.01975 | 0.03542 | 0.567 (4) |
| Random forest | 11 features + 4 lags | RandomForest_ext | 0.01970 | 0.03540 | 0.583 (3) |
| Random forest | 13 features + 4 lags | RandomForest | 0.01971 | 0.03540 | 0.615 (3) |
| SVR | 6 features + 4 lags | SVR_base | 0.01953 | 0.03676 | 0.404 (4) |
| SVR | 11 features + 4 lags | SVR_ext | 0.01952 | 0.03549 | 0.624 (4) |
| SVR | 13 features + 4 lags | SVR | 0.01942 | 0.03557 | 0.516 (4) |

## 2. Refit on targets up to 2017 (fixed epoch budget = best validation epoch of the original run + 1)

| model | fixed origin: test MAE (mean of seeds) | seed sd | seed-avg MAE | refit: test MAE (mean of seeds) | seed sd | seed-avg MAE | refit minus fixed (seed-avg) | DM p | years refit better | ex-COVID fixed / refit | COVID fixed / refit | best epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STGNN | 0.03521 | 0.00007 | 0.03517 | 0.03504 | 0.00008 | 0.03500 | -0.00017 | 0.629 | 5 | 0.0221 / 0.0221 | 0.0678 / 0.0673 | [114, 75, 289, 65, 102] |
| GRU | 0.03646 | 0.00035 | 0.03625 | 0.03680 | 0.00031 | 0.03657 | +0.00032 | 0.406 | 4 | 0.0222 / 0.0223 | 0.0714 / 0.0723 | [21, 26, 20, 34, 31] |
| LSTM | 0.03634 | 0.00006 | 0.03623 | 0.03671 | 0.00033 | 0.03639 | +0.00016 | 0.287 | 4 | 0.0222 / 0.0223 | 0.0712 / 0.0716 | [10, 15, 8, 24, 25] |
| Random forest | 0.03549 | 0.00002 | 0.03549 | 0.03549 | 0.00002 | 0.03548 | -0.00000 | 0.989 | 4 | 0.0226 / 0.0224 | 0.0676 / 0.0682 | config (3, 10, 0.6) |
| AR(1) pooled | 0.03811 | - | 0.03811 | 0.03800 | - | 0.03800 | -0.00011 | 0.808 | 5 | 0.0241 / 0.0238 | 0.0730 / 0.0735 | coef [0.285, 0.0269] |

Per-year MAE, fixed origin -> refit (seed-averaged):

| model | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|
| STGNN | 0.0150 -> 0.0149 | 0.0188 -> 0.0184 | 0.0935 -> 0.0918 | 0.0420 -> 0.0428 | 0.0333 -> 0.0339 | 0.0258 -> 0.0255 | 0.0177 -> 0.0176 |
| GRU | 0.0137 -> 0.0140 | 0.0187 -> 0.0187 | 0.0934 -> 0.0930 | 0.0494 -> 0.0516 | 0.0360 -> 0.0365 | 0.0259 -> 0.0257 | 0.0166 -> 0.0165 |
| LSTM | 0.0143 -> 0.0144 | 0.0188 -> 0.0187 | 0.0936 -> 0.0935 | 0.0489 -> 0.0497 | 0.0358 -> 0.0362 | 0.0257 -> 0.0256 | 0.0167 -> 0.0166 |
| Random forest | 0.0152 -> 0.0145 | 0.0195 -> 0.0193 | 0.0921 -> 0.0921 | 0.0432 -> 0.0442 | 0.0332 -> 0.0333 | 0.0277 -> 0.0275 | 0.0175 -> 0.0174 |
| AR(1) pooled | 0.0168 -> 0.0164 | 0.0225 -> 0.0219 | 0.0963 -> 0.0951 | 0.0498 -> 0.0518 | 0.0329 -> 0.0329 | 0.0294 -> 0.0291 | 0.0191 -> 0.0187 |

Refit models compared with each other (seed-averaged, DM vs refit STGNN):

- refit STGNN vs refit GRU: diff -0.00157, p 0.301, STGNN better in 5 years
- refit STGNN vs refit LSTM: diff -0.00140, p 0.247, STGNN better in 5 years
- refit STGNN vs refit Random forest: diff -0.00049, p 0.270, STGNN better in 4 years
- refit STGNN vs refit AR(1) pooled: diff -0.00300, p 0.057, STGNN better in 6 years

## 3. Expanding-window and Mondrian split CP for all seven models (90%)

| model | split cov | expanding cov | expanding 2020 | ACI cov | Mondrian cov | Mondrian low/med/high | split low/med/high |
|---|---|---|---|---|---|---|---|
| STGNN | 0.788 (file 0.786) | 0.835 | 0.231 | 0.842 | 0.747 | 0.616 / 0.757 / 0.869 | 0.862 / 0.783 / 0.719 |
| LSTM | 0.769 (file 0.769) | 0.825 | 0.225 | 0.840 | 0.734 | 0.607 / 0.736 / 0.859 | 0.843 / 0.752 / 0.712 |
| GRU | 0.776 (file 0.776) | 0.828 | 0.214 | 0.842 | 0.729 | 0.602 / 0.729 / 0.857 | 0.845 / 0.757 / 0.724 |
| Gradient boosting | 0.795 (file 0.795) | 0.831 | 0.187 | 0.842 | 0.754 | 0.625 / 0.764 / 0.874 | 0.869 / 0.793 / 0.724 |
| Random forest | 0.795 (file 0.795) | 0.834 | 0.236 | 0.841 | 0.737 | 0.604 / 0.738 / 0.869 | 0.855 / 0.795 / 0.735 |
| AR(1) pooled | 0.774 (file 0.774) | 0.827 | 0.231 | 0.838 | 0.738 | 0.625 / 0.740 / 0.848 | 0.841 / 0.764 / 0.717 |
| Rolling ARIMA with drift | 0.790 (file 0.790) | 0.834 | 0.286 | 0.840 | 0.743 | 0.630 / 0.755 / 0.845 | 0.831 / 0.800 / 0.738 |

Re-run check: max abs difference between conformal_results.csv and conformal_results.csv over all numeric cells: 0.00e+00
