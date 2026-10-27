# Missing values and imputation

Panel: 182 countries x 29 years (1996-2024) = 5278 country-years per feature. Real GDP and population are complete (countries with gaps were dropped).

| Feature | raw missing cells | filled by <=3-year interpolation | filled by cross-country yearly median | share of country-years imputed |
|---|---|---|---|---|
| inflation (CPI, annual %) | 464 | 104 | 360 | 8.8% |
| employment-to-population ratio (%) | 355 | 7 | 348 | 6.7% |

Imputed share of country-years by size tercile:

| group | inflation | employment ratio |
|---|---|---|
| small | 15.5% | 19.7% |
| medium | 8.9% | 0.2% |
| large | 1.9% | 0.2% |

Imputed share of country-years by income(GDPpc proxy) tercile:

| group | inflation | employment ratio |
|---|---|---|
| low | 10.5% | 3.6% |
| middle | 8.0% | 6.8% |
| high | 7.9% | 9.8% |

Test samples whose input window contains at least one imputed inflation or employment value: 12.7% (162 of 1274).

Test MAE on samples with a complete window versus samples containing imputed values:

| Model | complete windows | windows with imputed values | all |
|---|---|---|---|
| STGNN | 0.0337 | 0.0452 | 0.0352 |
| STGNN shuffled graph | 0.0337 | 0.0452 | 0.0352 |
| LSTM | 0.0347 | 0.0468 | 0.0362 |
| GRU | 0.0347 | 0.0466 | 0.0363 |
| GBM_win | 0.0344 | 0.0456 | 0.0358 |
| RandomForest_win5 | 0.0342 | 0.0446 | 0.0355 |
| SVR_win | 0.0351 | 0.0473 | 0.0366 |
| AR1_pooled | 0.0367 | 0.0480 | 0.0381 |
| Persistence(level) | 0.0469 | 0.0565 | 0.0482 |

If the models were relying on imputed values, the ranking of models would change on the complete-window subset; the subset ranking is shown above.