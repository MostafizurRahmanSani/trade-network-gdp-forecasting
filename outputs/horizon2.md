# Publication-lag scenario: forecasting two years ahead of the last observed GDP (targets 2019-2024)

h1 = main setting (window ending in the year before the target), h2 = window ending two years before the target. Same target years and countries; MAE in log growth points; neural rows are 5-seed means of predictions.

```
                        h1_all  h2_all  change  h1_exCOVID  h2_exCOVID
model                                                                 
STGNN_shuffled          0.0385  0.0396  0.0288      0.0239      0.0259
STGNN                   0.0385  0.0397  0.0290      0.0239      0.0259
MTGNN_trade             0.0392  0.0401  0.0233      0.0241      0.0265
Random forest           0.0389  0.0402  0.0357      0.0245      0.0267
LSTM                    0.0399  0.0402  0.0090      0.0242      0.0268
GRU                     0.0400  0.0403  0.0084      0.0243      0.0268
Ridge                   0.0402  0.0404  0.0035      0.0237      0.0268
SVR                     0.0403  0.0405  0.0040      0.0237      0.0267
GBM                     0.0392  0.0406  0.0360      0.0245      0.0272
RW drift                0.0418  0.0421  0.0071      0.0280      0.0287
AR(1) pooled (direct)   0.0417  0.0424  0.0179      0.0260      0.0299
ARIMA rolling (2-step)  0.0454  0.0459  0.0118      0.0274      0.0356
No-change               0.0500  0.0500  0.0000      0.0425      0.0425
```

Reading: if the last year's GDP is not available when the forecast is made, every model loses accuracy, the ranking among the learned models is essentially unchanged, and the no-change forecast is unaffected by construction.