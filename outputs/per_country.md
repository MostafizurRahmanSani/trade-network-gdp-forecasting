# Per-country test MAE (2018-24, seed-mean predictions)

STGNN beats AR(1) in 138 of 182 countries and no-change in 151; it beats LSTM in 132. Real vs shuffled graph: STGNN better in 68 countries, worse in 114 (mean |difference| 0.00004).

Largest gains over AR(1):

```
       STGNN  STGNN_shuffled    LSTM  GBM_win  AR1_pooled  Persistence(level)  STGNN-AR1
iso3                                                                                    
MAC   0.2701          0.2701  0.2770   0.2772      0.3218              0.2745    -0.0517
LBY   0.1486          0.1486  0.1508   0.1502      0.1662              0.1423    -0.0176
UKR   0.0715          0.0715  0.0759   0.0776      0.0863              0.0799    -0.0148
LCA   0.0867          0.0867  0.0933   0.0853      0.1002              0.0957    -0.0135
DEU   0.0201          0.0200  0.0222   0.0143      0.0336              0.0190    -0.0135
JPN   0.0172          0.0172  0.0188   0.0206      0.0307              0.0150    -0.0134
FIN   0.0149          0.0149  0.0165   0.0143      0.0280              0.0140    -0.0130
FRA   0.0248          0.0248  0.0271   0.0283      0.0374              0.0333    -0.0126
```

Largest losses versus AR(1):

```
       STGNN  STGNN_shuffled    LSTM  GBM_win  AR1_pooled  Persistence(level)  STGNN-AR1
iso3                                                                                    
GUY   0.2022          0.2021  0.1871   0.1929      0.1708              0.2549     0.0313
TLS   0.1571          0.1571  0.1542   0.1587      0.1376              0.1535     0.0195
SDN   0.1183          0.1184  0.1094   0.1196      0.0989              0.0877     0.0194
VEN   0.1485          0.1486  0.1410   0.1478      0.1342              0.1538     0.0143
FJI   0.0753          0.0754  0.0752   0.0719      0.0672              0.0806     0.0081
KAZ   0.0230          0.0230  0.0203   0.0194      0.0171              0.0402     0.0059
MNG   0.0307          0.0307  0.0295   0.0265      0.0251              0.0518     0.0056
MOZ   0.0205          0.0205  0.0203   0.0165      0.0154              0.0300     0.0051
```