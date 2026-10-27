# Ablation around the final model (final node feature set, all 13 edge features, final hidden size / weight decay, 5 seeds; validation MAE)

Each row changes exactly one component of the final STGNN (reference row). Scores are VALIDATION MAE (mean and sd over 5 seeds); the difference to the reference is paired by seed (paired t-test over 5 seeds, uncorrected for the 25 comparisons). Test MAE is deliberately not shown for variants; the final model's test MAE is reported in final_tables.md.

| Variant | hidden | what changes | val MAE | sd | diff vs final | p | parameters |
|---|---|---|---|---|---|---|---|
| final-tlast | 16 | no temporal integration: last input year only | 0.01923 | 0.00005 | -0.00052 | 0.000348 | 4451 |
| final-k1 | 16 | skip connection from the raw last-year features | 0.01925 | 0.00012 | -0.00050 | 0.00129 | 6195 |
| final-d2 | 16 | dropout 0.2 | 0.01969 | 0.00006 | -0.00006 | 0.0288 | 6083 |
| final-evec | 16 | per-channel gate from all 13 edge features | 0.01971 | 0.00012 | -0.00004 | 0.636 | 6289 |
| final-evecmlp | 16 | per-channel gate, 13->h->h MLP | 0.01973 | 0.00011 | -0.00003 | 0.678 | 6833 |
| final-l1 | 16 | 1 message-passing layer | 0.01973 | 0.00018 | -0.00003 | 0.793 | 4058 |
| final-eattn | 16 | attention over incoming edges (uses edge features) | 0.01974 | 0.00010 | -0.00002 | 0.246 | 5931 |
| final (reference) | 16 | final design (2 message-passing layers, GRU update, learned scalar edge weight, mean aggregation, L2-normalised, 1-layer temporal GRU) | 0.01975 | 0.00009 | +0.00000 | - | 6083 |
| final-emlp4 | 16 | edge weight MLP 13->4->1 | 0.01976 | 0.00002 | +0.00001 | 0.882 | 5963 |
| final-amax | 16 | max aggregation | 0.01976 | 0.00008 | +0.00001 | 0.706 | 6083 |
| final (reference) | 32 | hidden size 32 | 0.01977 | 0.00008 | +0.00002 | 0.743 | 22915 |
| final-l0 | 16 | no message passing (0 layers): same temporal GRU on per-country features | 0.01977 | 0.00009 | +0.00002 | 0.736 | 2033 |
| final-l3 | 16 | 3 message-passing layers | 0.01982 | 0.00012 | +0.00007 | 0.402 | 8108 |
| final (reference) | 64 | hidden size 64 | 0.01983 | 0.00005 | +0.00007 | 0.177 | 88835 |
| final-econcat | 16 | edge features concatenated into the message | 0.01984 | 0.00008 | +0.00008 | 0.261 | 6257 |
| final-euniform | 16 | uniform edge weights (no edge features) | 0.01985 | 0.00011 | +0.00010 | 0.28 | 5841 |
| final-emlp32 | 16 | edge weight MLP 13->32->1 | 0.01985 | 0.00009 | +0.00010 | 0.19 | 6803 |
| final-emlp16 | 16 | edge weight MLP 13->16->1 | 0.01989 | 0.00005 | +0.00014 | 0.0565 | 6323 |
| final-elinear | 16 | edge weight = sigmoid(linear(13 features)) | 0.01989 | 0.00017 | +0.00014 | 0.175 | 5869 |
| final-asum | 16 | sum aggregation | 0.01994 | 0.00012 | +0.00018 | 0.0508 | 6083 |
| final-cadd | 16 | additive update instead of GRU update | 0.02000 | 0.00004 | +0.00024 | 0.00696 | 3363 |
| final-cconcat | 16 | concatenation update | 0.02000 | 0.00007 | +0.00024 | 0.0015 | 3875 |
| final-g2 | 16 | 2-layer temporal GRU | 0.02064 | 0.00013 | +0.00089 | 3.63e-05 | 7715 |
| final-tmean | 16 | temporal mean instead of GRU | 0.02139 | 0.00016 | +0.00164 | 8.89e-05 | 4451 |
| final-g3 | 16 | 3-layer temporal GRU | 0.02241 | 0.00098 | +0.00265 | 0.00342 | 9347 |
| final-n0 | 16 | no L2 normalisation of node states | 0.02249 | 0.00077 | +0.00274 | 0.00108 | 6083 |

Seed-to-seed sd of the reference validation MAE: 0.00009. Differences smaller than about 2 x that value (0.0001-0.0002) are within noise.