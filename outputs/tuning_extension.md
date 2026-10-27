# Extended tuning check (neural families)

Around each family's selected configuration (feature set, hidden, weight decay from the original grid) I added learning rate 3e-4 and 3e-3 (selected: 1e-3), hidden size 8 (families selected with 16) or 64 (families selected with 32) and weight decay 1e-2. Each extra configuration: 5 seeds, validation MAE; test results play no role.
Decision rule (fixed beforehand): adopt an extra configuration only if its mean validation MAE beats the selected one by more than 0.0003.

| Family | selected (feat, h, wd) | val | lr 3e-4 | lr 3e-3 | hidden 8/64 | wd 1e-2 | best extra | improvement | adopt |
|---|---|---|---|---|---|---|---|---|---|
| STGNN | base, 16, 1e-3 | 0.01972 | 0.01974 | 0.01979 | 0.01955 | 0.02346 | hidden 8 | +0.00016 | no |
| STGCN | base, 16, 1e-3 | 0.02030 | 0.02029 | 0.02037 | 0.02024 | 0.02377 | hidden 8 | +0.00006 | no |
| STGAT | base, 16, 0 | 0.02015 | 0.02015 | 0.02004 | 0.02003 | 0.02357 | hidden 8 | +0.00011 | no |
| DCRNN | base, 16, 1e-3 | 0.02002 | 0.02006 | 0.01997 | 0.01996 | 0.02174 | hidden 8 | +0.00006 | no |
| GWN | base, 32, 1e-3 | 0.02048 | 0.02038 | 0.02085 | 0.02050 | 0.02301 | lr 3e-4 | +0.00011 | no |
| GNN | base, 32, 1e-3 | 0.01921 | 0.01930 | 0.01924 | 0.01923 | 0.02045 | hidden 64 | -0.00002 | no |
| GCN | base, 16, 1e-3 | 0.02301 | 0.02304 | 0.02305 | 0.02325 | 0.02408 | lr 3e-4 | -0.00003 | no |
| GAT | base, 16, 1e-3 | 0.01943 | 0.01950 | 0.01939 | 0.01939 | 0.02173 | lr 3e-3 | +0.00004 | no |
| GRU | base, 16, 1e-3 | 0.01943 | 0.01955 | 0.01943 | 0.01956 | 0.01995 | lr 3e-3 | +0.00001 | no |
| LSTM | base, 16, 1e-3 | 0.01950 | 0.01964 | 0.01950 | 0.01946 | 0.02031 | hidden 8 | +0.00003 | no |
| RNN | base, 32, 1e-3 | 0.01920 | 0.01918 | 0.01919 | 0.01910 | 0.01959 | hidden 64 | +0.00010 | no |
| BiLSTM | base, 32, 1e-3 | 0.01944 | 0.01950 | 0.01943 | 0.01933 | 0.02076 | hidden 64 | +0.00011 | no |
| Transformer | base, 16, 1e-3 | 0.01934 | 0.01950 | 0.01925 | 0.01920 | 0.01925 | hidden 8 | +0.00014 | no |
| TFT | base, 32, 1e-3 | 0.01903 | 0.01918 | 0.01903 | 0.01912 | 0.01910 | lr 3e-3 | -0.00000 | no |
| MTGNN | base, 32, 1e-3 | 0.01937 | 0.01988 | 0.01926 | 0.01928 | 0.01960 | lr 3e-3 | +0.00011 | no |
| MTGNN_trade | base, 32, 1e-3 | 0.01939 | 0.01991 | 0.01936 | 0.01931 | 0.01940 | hidden 64 | +0.00008 | no |

Families adopting a new configuration: none.