# Sample-selection sensitivity (targets 2018-2023)

Main sample: 182 economies (trained and evaluated there). Extended sample: 189 economies (the 7 additions have minor gaps in 1994 or 2024 only).

```
                main_sample_182  extended_sample_189  only_182_original  only_7_added
model                                                                                
No-change                0.0499               0.0495             0.0499        0.0400
AR(1) pooled             0.0413               0.0412             0.0412        0.0412
STGNN                    0.0381               0.0381             0.0381        0.0393
STGNN_shuffled           0.0381               0.0382             0.0381        0.0394
LSTM                     0.0395               0.0395             0.0396        0.0386
GRU                      0.0395               0.0396             0.0396        0.0379
```

Ranking of the models on the main sample: STGNN_shuffled, STGNN, LSTM, GRU, AR(1) pooled, No-change.
Ranking on the extended sample: STGNN, STGNN_shuffled, LSTM, GRU, AR(1) pooled, No-change.

Economies with long gaps (Afghanistan, Syria, Yemen, South Sudan, Eritrea, North Korea, small territories) cannot be added, so the sensitivity covers only part of the sample-selection issue.