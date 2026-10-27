# Additional analyses of the conformal results (computed from saved outputs)

## 1. Conformal results without 2020-2021 (mean of the five non-COVID test years)

| alpha | model | method | coverage all | coverage excl. 2020-21 | Winkler all | Winkler excl. 2020-21 |
|---|---|---|---|---|---|---|
| 0.1 | STGNN (5-seed mean) | split-CP | 0.786 | 0.910 | 0.359 | 0.214 |
| 0.1 | STGNN (5-seed mean) | ACI | 0.842 | 0.953 | 0.378 | 0.234 |
| 0.1 | STGNN (5-seed mean) | rolling-CP | 0.846 | 0.957 | 0.387 | 0.241 |
| 0.1 | LSTM (5-seed mean) | split-CP | 0.769 | 0.905 | 0.372 | 0.214 |
| 0.1 | LSTM (5-seed mean) | ACI | 0.840 | 0.957 | 0.381 | 0.234 |
| 0.1 | LSTM (5-seed mean) | rolling-CP | 0.841 | 0.957 | 0.390 | 0.242 |
| 0.1 | GRU (5-seed mean) | split-CP | 0.776 | 0.912 | 0.369 | 0.211 |
| 0.1 | GRU (5-seed mean) | ACI | 0.842 | 0.959 | 0.379 | 0.231 |
| 0.1 | GRU (5-seed mean) | rolling-CP | 0.846 | 0.962 | 0.387 | 0.240 |
| 0.1 | GBM | split-CP | 0.795 | 0.915 | 0.360 | 0.219 |
| 0.1 | GBM | ACI | 0.842 | 0.953 | 0.384 | 0.239 |
| 0.1 | GBM | rolling-CP | 0.845 | 0.956 | 0.390 | 0.246 |
| 0.1 | RandomForest | split-CP | 0.795 | 0.919 | 0.355 | 0.216 |
| 0.1 | RandomForest | ACI | 0.841 | 0.951 | 0.375 | 0.236 |
| 0.1 | RandomForest | rolling-CP | 0.849 | 0.956 | 0.380 | 0.241 |
| 0.1 | AR(1) pooled | split-CP | 0.774 | 0.912 | 0.388 | 0.207 |
| 0.1 | AR(1) pooled | ACI | 0.838 | 0.953 | 0.396 | 0.233 |
| 0.1 | AR(1) pooled | rolling-CP | 0.845 | 0.959 | 0.404 | 0.239 |
| 0.1 | ARIMA rolling | split-CP | 0.751 | 0.890 | 0.419 | 0.223 |
| 0.1 | ARIMA rolling | ACI | 0.829 | 0.953 | 0.410 | 0.245 |
| 0.1 | ARIMA rolling | rolling-CP | 0.839 | 0.957 | 0.415 | 0.261 |
| 0.1 | ARIMA rolling, drift | split-CP | 0.790 | 0.919 | 0.386 | 0.218 |
| 0.1 | ARIMA rolling, drift | ACI | 0.840 | 0.954 | 0.395 | 0.242 |
| 0.1 | ARIMA rolling, drift | rolling-CP | 0.845 | 0.959 | 0.402 | 0.251 |
| 0.1 | GBM-quantile | native quantile | 0.809 | 0.927 | 0.332 | 0.194 |
| 0.1 | Linear-quantile | native quantile | 0.811 | 0.942 | 0.355 | 0.213 |
| 0.05 | STGNN (5-seed mean) | split-CP | 0.882 | 0.959 | 0.512 | 0.321 |
| 0.05 | STGNN (5-seed mean) | ACI | 0.897 | 0.975 | 0.565 | 0.353 |
| 0.05 | STGNN (5-seed mean) | rolling-CP | 0.900 | 0.975 | 0.553 | 0.347 |
| 0.05 | LSTM (5-seed mean) | split-CP | 0.859 | 0.954 | 0.538 | 0.317 |
| 0.05 | LSTM (5-seed mean) | ACI | 0.892 | 0.977 | 0.569 | 0.353 |
| 0.05 | LSTM (5-seed mean) | rolling-CP | 0.889 | 0.977 | 0.565 | 0.344 |
| 0.05 | GRU (5-seed mean) | split-CP | 0.859 | 0.955 | 0.535 | 0.313 |
| 0.05 | GRU (5-seed mean) | ACI | 0.893 | 0.978 | 0.567 | 0.347 |
| 0.05 | GRU (5-seed mean) | rolling-CP | 0.887 | 0.977 | 0.568 | 0.341 |
| 0.05 | GBM | split-CP | 0.870 | 0.954 | 0.525 | 0.327 |
| 0.05 | GBM | ACI | 0.899 | 0.976 | 0.563 | 0.355 |
| 0.05 | GBM | rolling-CP | 0.899 | 0.976 | 0.557 | 0.354 |
| 0.05 | RandomForest | split-CP | 0.871 | 0.957 | 0.520 | 0.322 |
| 0.05 | RandomForest | ACI | 0.895 | 0.974 | 0.556 | 0.353 |
| 0.05 | RandomForest | rolling-CP | 0.898 | 0.974 | 0.542 | 0.349 |
| 0.05 | AR(1) pooled | split-CP | 0.855 | 0.955 | 0.558 | 0.308 |
| 0.05 | AR(1) pooled | ACI | 0.889 | 0.979 | 0.604 | 0.360 |
| 0.05 | AR(1) pooled | rolling-CP | 0.882 | 0.979 | 0.600 | 0.346 |
| 0.05 | ARIMA rolling | split-CP | 0.826 | 0.942 | 0.636 | 0.323 |
| 0.05 | ARIMA rolling | ACI | 0.894 | 0.974 | 0.579 | 0.351 |
| 0.05 | ARIMA rolling | rolling-CP | 0.893 | 0.977 | 0.591 | 0.358 |
| 0.05 | ARIMA rolling, drift | split-CP | 0.850 | 0.946 | 0.575 | 0.317 |
| 0.05 | ARIMA rolling, drift | ACI | 0.896 | 0.973 | 0.569 | 0.347 |
| 0.05 | ARIMA rolling, drift | rolling-CP | 0.901 | 0.973 | 0.562 | 0.351 |
| 0.05 | GBM-quantile | native quantile | 0.872 | 0.956 | 0.496 | 0.287 |
| 0.05 | Linear-quantile | native quantile | 0.882 | 0.967 | 0.504 | 0.316 |

Year-clustered DM test on the yearly mean Winkler score (split CP, 90%, STGNN minus model; HLN, t(6)):

| model | mean diff | p | years STGNN lower |
|---|---|---|---|
| LSTM (5-seed mean) | -0.0129 | 0.478 | 2 |
| GRU (5-seed mean) | -0.0103 | 0.595 | 2 |
| GBM | -0.0013 | 0.797 | 5 |
| RandomForest | +0.0040 | 0.589 | 5 |
| AR(1) pooled | -0.0290 | 0.310 | 3 |
| ARIMA rolling | -0.0608 | 0.497 | 4 |
| ARIMA rolling, drift | -0.0277 | 0.532 | 4 |

alpha 0.1 split-CP: coverage 0.7512-0.7951; 2018: 0.9505-0.9670; 2019: 0.9341-0.9560; 2020: 0.2473-0.3901; 2021: 0.4176-0.7033; 2022: 0.7308-0.8571; 2023: 0.8736-0.9176; 2024: 0.9286-0.9615

alpha 0.1 normalised-CP: coverage 0.7355-0.7739; 2018: 0.9341-0.9451; 2019: 0.8846-0.9231; 2020: 0.1868-0.2802; 2021: 0.4121-0.6319; 2022: 0.7253-0.8681; 2023: 0.8791-0.9396; 2024: 0.9396-0.9780

alpha 0.1 ACI: coverage 0.8289-0.8422; 2018: 0.9505-0.9615; 2019: 0.9066-0.9341; 2020: 0.1868-0.3352; 2021: 0.7033-0.9451; 2022: 0.9341-0.9451; 2023: 0.9670-0.9725; 2024: 0.9780-0.9835

alpha 0.1 rolling-CP: coverage 0.8391-0.8485; 2018: 0.9505-0.9670; 2019: 0.9066-0.9341; 2020: 0.1868-0.3132; 2021: 0.7747-0.9505; 2022: 0.9396-0.9615; 2023: 0.9725-0.9835; 2024: 0.9725-0.9835

alpha 0.05 split-CP: coverage 0.8257-0.8823; 2018: 0.9725-0.9835; 2019: 0.9560-0.9725; 2020: 0.4396-0.5330; 2021: 0.5604-0.8626; 2022: 0.8516-0.9121; 2023: 0.9396-0.9560; 2024: 0.9725-0.9780

alpha 0.05 normalised-CP: coverage 0.8100-0.8509; 2018: 0.9615-0.9780; 2019: 0.9670-0.9780; 2020: 0.3297-0.3846; 2021: 0.5165-0.7857; 2022: 0.8681-0.9231; 2023: 0.9451-0.9780; 2024: 0.9670-0.9890

alpha 0.05 ACI: coverage 0.8885-0.8987; 2018: 0.9670-0.9835; 2019: 0.9505-0.9670; 2020: 0.3516-0.4945; 2021: 0.8956-0.9890; 2022: 0.9615-0.9780; 2023: 0.9780-0.9890; 2024: 0.9835-0.9945

alpha 0.05 rolling-CP: coverage 0.8823-0.9011; 2018: 0.9725-0.9835; 2019: 0.9505-0.9670; 2020: 0.3187-0.4945; 2021: 0.8956-0.9835; 2022: 0.9670-0.9780; 2023: 0.9780-0.9890; 2024: 0.9780-0.9890

## 2. Test-period outcomes

Median real GDP growth over the 1,274 test country-years: 0.0303 log points; median excluding 2020: 0.0353.

## 3. Self-loops and duplicate edges in the randomised graph controls (years 1996-2023, per seed; share of all edges)

| control | seed | edges | self-loops | duplicate edges |
|---|---|---|---|---|
| shuffle | 42 | 60814 | 324 (0.53%) | 8407 (13.82%) |
| random | 42 | 60814 | 351 (0.58%) | 1966 (3.23%) |
| shuffle | 1 | 60814 | 318 (0.52%) | 8381 (13.78%) |
| random | 1 | 60814 | 354 (0.58%) | 1968 (3.24%) |
| shuffle | 2 | 60814 | 332 (0.55%) | 8396 (13.81%) |
| random | 2 | 60814 | 357 (0.59%) | 2005 (3.30%) |
| shuffle | 3 | 60814 | 327 (0.54%) | 8371 (13.76%) |
| random | 3 | 60814 | 365 (0.60%) | 1995 (3.28%) |
| shuffle | 4 | 60814 | 311 (0.51%) | 8396 (13.81%) |
| random | 4 | 60814 | 348 (0.57%) | 2003 (3.29%) |
