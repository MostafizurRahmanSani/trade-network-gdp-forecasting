# IMF WEO comparison (real GDP growth, log points; IMF growth % converted with log(1+g/100))

## target years 2018-2023 (IMF October forecast available); country-years with an IMF October forecast: 1061 country-years

| Forecast | MAE | RMSE |
|---|---|---|
| IMF Fall(t) next-year | 0.0350 | 0.0659 |
| STGNN | 0.0381 | 0.0708 |
| RandomForest_win5 | 0.0385 | 0.0709 |
| GBM_win | 0.0388 | 0.0715 |
| LSTM | 0.0394 | 0.0717 |
| GRU | 0.0394 | 0.0715 |
| SVR_win | 0.0398 | 0.0729 |
| AR1_pooled | 0.0411 | 0.0738 |
| RW_drift | 0.0414 | 0.0739 |
| ARIMA_rolling | 0.0447 | 0.0792 |
| Persistence(level) | 0.0501 | 0.0748 |

## ex-COVID 2018-2019 and 2022-2023; country-years with an IMF October forecast: 707 country-years

| Forecast | MAE | RMSE |
|---|---|---|
| IMF Fall(t) next-year | 0.0215 | 0.0443 |
| SVR_win | 0.0232 | 0.0510 |
| STGNN | 0.0234 | 0.0500 |
| GRU | 0.0237 | 0.0495 |
| LSTM | 0.0238 | 0.0500 |
| GBM_win | 0.0239 | 0.0512 |
| RandomForest_win5 | 0.0241 | 0.0505 |
| AR1_pooled | 0.0254 | 0.0503 |
| ARIMA_rolling | 0.0268 | 0.0512 |
| RW_drift | 0.0275 | 0.0534 |
| Persistence(level) | 0.0428 | 0.0618 |

## COVID 2020-2021; country-years with an IMF October forecast: 354 country-years

| Forecast | MAE | RMSE |
|---|---|---|
| IMF Fall(t) next-year | 0.0621 | 0.0953 |
| Persistence(level) | 0.0647 | 0.0957 |
| RandomForest_win5 | 0.0673 | 0.0999 |
| STGNN | 0.0674 | 0.1002 |
| GBM_win | 0.0684 | 0.1005 |
| RW_drift | 0.0692 | 0.1033 |
| LSTM | 0.0707 | 0.1020 |
| GRU | 0.0708 | 0.1020 |
| AR1_pooled | 0.0725 | 0.1061 |
| SVR_win | 0.0728 | 0.1037 |
| ARIMA_rolling | 0.0803 | 0.1165 |

Per-year MAE (common sample):

| Forecast | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| STGNN | 0.0151 | 0.0190 | 0.0926 | 0.0423 | 0.0339 | 0.0255 |
| LSTM | 0.0144 | 0.0190 | 0.0926 | 0.0488 | 0.0363 | 0.0253 |
| GRU | 0.0138 | 0.0189 | 0.0924 | 0.0492 | 0.0365 | 0.0255 |
| GBM_win | 0.0154 | 0.0196 | 0.0929 | 0.0439 | 0.0344 | 0.0262 |
| RandomForest_win5 | 0.0153 | 0.0197 | 0.0911 | 0.0435 | 0.0338 | 0.0275 |
| SVR_win | 0.0146 | 0.0191 | 0.0940 | 0.0517 | 0.0340 | 0.0250 |
| AR1_pooled | 0.0167 | 0.0226 | 0.0951 | 0.0498 | 0.0335 | 0.0287 |
| ARIMA_rolling | 0.0173 | 0.0193 | 0.0854 | 0.0753 | 0.0426 | 0.0279 |
| Persistence(level) | 0.0376 | 0.0363 | 0.0688 | 0.0606 | 0.0562 | 0.0409 |
| RW_drift | 0.0197 | 0.0243 | 0.0970 | 0.0413 | 0.0374 | 0.0284 |
| IMF Fall(t) next-year | 0.0156 | 0.0187 | 0.0902 | 0.0340 | 0.0308 | 0.0208 |

Notes: the Fall(t) vintage predates the full-year-t outturn, while the models use full-year-t data (through year t), so the information sets differ. Target 2024 is excluded because the historical-forecast file ends with the Fall 2022 vintage.