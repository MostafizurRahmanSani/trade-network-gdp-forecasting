# Additional tables 

## 1. Training stability and checkpoint sensitivity (STGNN, base features, h16, wd 1e-3, 5 seeds)

Configurations: epochs 100/300/1000, gradient clipping (max-norm 1), cosine learning-rate schedule. Checkpoint rules: best-validation epoch, mean over +-5 epochs around it, average of the weights of the last 10 epochs, last epoch.

```
                seeds  val(best epoch)  test @ best-val epoch  test mean over +-5 epochs  test last-10 weight avg  test last epoch  sd of test MAE over last 50 epochs  sd of val MAE over last 50 epochs  mean best epoch  sd test@best across seeds
config                                                                                                                                                                                                                                               
e100                5          0.01975                0.03519                    0.03520                  0.03520          0.03521                             0.00005                            0.00007             78.2                    0.00005
e300                5          0.01972                0.03521                    0.03521                  0.03529          0.03530                             0.00002                            0.00003            129.0                    0.00007
e300_clip           5          0.01972                0.03521                    0.03521                  0.03529          0.03530                             0.00002                            0.00003            129.0                    0.00007
e300_cos            5          0.01970                0.03525                    0.03526                  0.03536          0.03536                             0.00000                            0.00000            132.2                    0.00006
e300_clip_cos       5          0.01970                0.03525                    0.03526                  0.03536          0.03536                             0.00000                            0.00000            132.2                    0.00006
e1000               5          0.01970                0.03523                    0.03523                  0.03534          0.03536                             0.00002                            0.00003            193.2                    0.00007
e1000_clip_cos      5          0.01970                0.03526                    0.03526                  0.03545          0.03545                             0.00000                            0.00000            166.6                    0.00007
```

## 2. Node/edge feature normalisation: training-period (global) vs per-year (5 seeds each, same configuration)

| Model | normalisation | seeds | val MAE | test MAE | sd |
|---|---|---|---|---|---|
| STGNN | training-period (used) | 5 | 0.0197 | 0.0352 | 0.0001 |
| STGNN | per-year | 5 | 0.0207 | 0.0359 | 0.0002 |
| LSTM | training-period (used) | 5 | 0.0195 | 0.0363 | 0.0001 |
| LSTM | per-year | 5 | 0.0198 | 0.0351 | 0.0001 |
| GRU | training-period (used) | 5 | 0.0194 | 0.0365 | 0.0003 |
| GRU | per-year | 5 | 0.0197 | 0.0351 | 0.0002 |

## 3. Capacity check: recurrent baselines with more parameters than the STGNN

| Model | hidden | parameters | seeds | val MAE | test MAE |
|---|---|---|---|---|---|
| LSTM (base) | 16 | 1553 | 5 | 0.0195 | 0.0363 |
| LSTM (base) | 48 | 10801 | 5 | 0.0196 | 0.0363 |
| LSTM (base) | 64 | 18497 | 5 | 0.0195 | 0.0366 |
| GRU (base) | 16 | 1169 | 5 | 0.0194 | 0.0365 |
| GRU (base) | 48 | 8113 | 5 | 0.0196 | 0.0364 |
| GRU (base) | 64 | 13889 | 5 | 0.0196 | 0.0365 |
| STGNN (base) | 16 | 4161 | 5 | 0.0197 | 0.0352 |

## 4. Parameter counts, run time and selected configuration (validation-selected, 5 seeds)

| Model | features | hidden | wd | parameters | seconds per run (CPU, 1 thread) |
|---|---|---|---|---|---|
| STGNN | base (6) | 16 | 1e-3 | 4161 | 41 |
| STGCN | base (6) | 16 | 1e-3 | 2698 | 79 |
| STGAT | base (6) | 16 | 0 | 4490 | 123 |
| DCRNN | base (6) | 16 | 1e-3 | 5738 | 62 |
| GWN | base (6) | 32 | 1e-3 | 11730 | 19 |
| GNN | base (6) | 32 | 1e-3 | 16338 | 18 |
| GCN | base (6) | 16 | 1e-3 | 1066 | 18 |
| GAT | base (6) | 16 | 1e-3 | 2858 | 23 |
| BonnigGAT | base (6) | 16 | 0 | 465 | 13 |
| BonnigGAT_edges | base (6) | 32 | 0 | 2337 | 19 |
| GRU | base (6) | 16 | 1e-3 | 1169 | 62 |
| LSTM | base (6) | 16 | 1e-3 | 1553 | 50 |
| RNN | base (6) | 32 | 1e-3 | 1313 | 35 |
| BiLSTM | base (6) | 32 | 1e-3 | 10305 | 58 |
| Transformer | base (6) | 16 | 1e-3 | 2449 | 66 |
| TFT | base (6) | 32 | 1e-3 | 31533 | 175 |
| MTGNN | base (6) | 32 | 1e-3 | 22337 | 48 |
| MTGNN_trade | base (6) | 32 | 1e-3 | 26433 | 57 |

Selection protocol (every neural family, validation MAE only): all models use the 6 base node features (the larger sets {ext: 11, partner: 13} were part of the first grid and are out of the final selection by author decision); candidate configurations hidden size {16, 32} x weight decay {0, 1e-3}, each with 5 seeds (42, 1, 2, 3, 4); the configuration with the lowest 5-seed mean validation MAE is selected; an extended check (learning rate 3e-4/3e-3, hidden size 8 or 64, weight decay 1e-2; rule: adopt only if validation improves by more than 0.0003) changed only the LSTM (hidden 8). Adam, learning rate 1e-3, up to 300 epochs, checkpoint with the lowest validation MAE. The seed controls weight initialisation, the mini-batch order of the recurrent baselines and the randomised graph controls; the spatio-temporal models visit training windows in chronological order; tabular baselines use random_state=0.
Message direction: edges run exporter -> importer (message from source i to destination j); spatial-only models (GNN, GCN, GAT, Bonnig-style GAT) use the graph of the last input year.