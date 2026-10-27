# Rolling ARIMA with drift

| version | order | val MAE | test MAE | ex-COVID | COVID | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | IMF sample all | IMF ex-COVID (2018-19, 2022-23) | IMF COVID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no drift (12_arima_rolling.py) | see 12_arima_rolling.py log | 0.02244 | 0.04138 | 0.02541 | 0.08131 | 0.0172 | 0.0190 | 0.0862 | 0.0764 | 0.0423 | 0.0286 | 0.0200 | 0.0447 | 0.0268 | 0.0803 |
| with drift (13_arima_drift.py) | (np.int64(1), np.int64(1), np.int64(0)) | 0.02189 | 0.03900 | 0.02382 | 0.07695 | 0.0164 | 0.0198 | 0.0931 | 0.0608 | 0.0384 | 0.0268 | 0.0177 | 0.0422 | 0.0253 | 0.0759 |
| CP protocol, no drift | (np.int64(1), np.int64(1), np.int64(1)) | 0.02307 | 0.04138 | 0.02541 | 0.08131 | 0.0172 | 0.0190 | 0.0862 | 0.0764 | 0.0423 | 0.0286 | 0.0200 | 0.0447 | 0.0268 | 0.0803 |
| CP protocol, with drift | (np.int64(1), np.int64(1), np.int64(0)) | 0.02391 | 0.03900 | 0.02382 | 0.07695 | 0.0164 | 0.0198 | 0.0931 | 0.0608 | 0.0384 | 0.0268 | 0.0177 | 0.0422 | 0.0253 | 0.0759 |

IMF October sample: 1061 country-years. Val MAE: main protocol targets 2015-17; CP protocol targets 2013-14.
