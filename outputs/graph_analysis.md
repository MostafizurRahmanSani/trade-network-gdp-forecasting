# Graph statistics: each exporter keeps its top-12 destinations by total trade value

```
 year  nodes  edges  density  mean_out  max_out  mean_in  max_in  isolated_in  in_degree_gini  top5_share_of_in_edges  reciprocity
 1996    182   2100    0.064    11.538     12.0   11.538   154.0           52           0.786                   0.320        0.175
 1997    182   2093    0.064    11.500     12.0   11.500   157.0           49           0.788                   0.324        0.173
 1998    182   2097    0.064    11.522     12.0   11.522   152.0           45           0.775                   0.317        0.178
 1999    182   2108    0.064    11.582     12.0   11.582   157.0           48           0.780                   0.322        0.184
 2000    182   2184    0.066    12.000     12.0   12.000   163.0           34           0.767                   0.318        0.179
 2001    182   2184    0.066    12.000     12.0   12.000   164.0           36           0.767                   0.311        0.171
 2002    182   2184    0.066    12.000     12.0   12.000   163.0           26           0.757                   0.309        0.179
 2003    182   2184    0.066    12.000     12.0   12.000   160.0           27           0.758                   0.309        0.175
 2004    182   2184    0.066    12.000     12.0   12.000   156.0           33           0.755                   0.298        0.180
 2005    182   2184    0.066    12.000     12.0   12.000   157.0           32           0.746                   0.297        0.190
 2006    182   2184    0.066    12.000     12.0   12.000   154.0           32           0.751                   0.290        0.179
 2007    182   2184    0.066    12.000     12.0   12.000   147.0           32           0.747                   0.282        0.181
 2008    182   2184    0.066    12.000     12.0   12.000   147.0           31           0.743                   0.273        0.177
 2009    182   2184    0.066    12.000     12.0   12.000   146.0           26           0.739                   0.284        0.185
 2010    182   2184    0.066    12.000     12.0   12.000   145.0           26           0.735                   0.275        0.180
 2011    182   2184    0.066    12.000     12.0   12.000   140.0           32           0.740                   0.273        0.179
 2012    182   2184    0.066    12.000     12.0   12.000   148.0           26           0.737                   0.270        0.184
 2013    182   2184    0.066    12.000     12.0   12.000   150.0           25           0.727                   0.270        0.188
 2014    182   2184    0.066    12.000     12.0   12.000   150.0           22           0.730                   0.269        0.185
 2015    182   2184    0.066    12.000     12.0   12.000   152.0           27           0.723                   0.271        0.189
 2016    182   2184    0.066    12.000     12.0   12.000   145.0           25           0.725                   0.268        0.188
 2017    182   2184    0.066    12.000     12.0   12.000   145.0           29           0.734                   0.271        0.183
 2018    182   2184    0.066    12.000     12.0   12.000   150.0           25           0.731                   0.274        0.192
 2019    182   2184    0.066    12.000     12.0   12.000   150.0           31           0.731                   0.276        0.182
 2020    182   2184    0.066    12.000     12.0   12.000   149.0           28           0.725                   0.274        0.190
 2021    182   2184    0.066    12.000     12.0   12.000   150.0           27           0.729                   0.270        0.187
 2022    182   2184    0.066    12.000     12.0   12.000   148.0           26           0.729                   0.264        0.191
 2023    182   2184    0.066    12.000     12.0   12.000   143.0           26           0.729                   0.262        0.189
```

The graph is sparse (density 0.066, about 11.9 out-edges per country) with a highly skewed in-degree (max 164, Gini 0.75): a few large importers receive most edges. The fully connected alternative has density 1.0.

Message-passing GRU weights (weight decay 0.001): layer1: ||W_message||=0.0000, ||W_state||=0.0184, ratio 0.001. W_message multiplies the aggregated neighbour message, W_state the node's own state; a ratio near 0 means the update ignores the neighbours.

# Learned edge weights, weight decay 0.001 (STGNN edge-weight network on the 2023 graph; weights in (0,1))

mean 0.50000, sd 0.000000, min 0.50000, max 0.50000; share above their mean: 0.00

Spearman correlation of the learned weight with each (standardised) edge feature:

```
lv    NaN
xs    NaN
ms    NaN
gs0   NaN
gs1   NaN
gs2   NaN
gs3   NaN
gs4   NaN
gs5   NaN
gs6   NaN
gs7   NaN
gs8   NaN
gs9   NaN
```

Spearman(weight, importer log GDP) = nan; Spearman(weight, exporter log GDP) = nan

Top 5 and bottom 5 edges by learned weight (exporter -> importer):

```
exporter importer           v   w
     ALB      ITA 1739732.245 0.5
     ALB      GRC  408596.922 0.5
     ALB      DEU  237293.209 0.5
     ALB      ESP  206831.252 0.5
     ALB      CHN  157482.072 0.5
     ALB      ITA 1739732.245 0.5
     ALB      GRC  408596.922 0.5
     ALB      DEU  237293.209 0.5
     ALB      ESP  206831.252 0.5
     ALB      CHN  157482.072 0.5
```

Message-passing GRU weights (weight decay 0.0): layer1: ||W_message||=4.8117, ||W_state||=4.1918, ratio 1.148. W_message multiplies the aggregated neighbour message, W_state the node's own state; a ratio near 0 means the update ignores the neighbours.

# Learned edge weights, weight decay 0.0 (STGNN edge-weight network on the 2023 graph; weights in (0,1))

mean 0.51031, sd 0.008405, min 0.48931, max 0.54282; share above their mean: 0.49

Spearman correlation of the learned weight with each (standardised) edge feature:

```
lv     0.411
xs     0.249
ms     0.511
gs0   -0.252
gs1    0.616
gs2    0.120
gs3    0.310
gs4    0.243
gs5    0.211
gs6   -0.071
gs7    0.061
gs8   -0.045
gs9   -0.033
```

Spearman(weight, importer log GDP) = -0.107; Spearman(weight, exporter log GDP) = 0.347

Top 5 and bottom 5 edges by learned weight (exporter -> importer):

```
exporter importer          v      w
     HTI      USA 844427.436 0.5428
     BEN      BGD 391460.601 0.5358
     TJK      IRN  99706.245 0.5328
     MLI      BGD 205452.713 0.5321
     CMR      BGD 211255.292 0.5321
     KIR      CZE    373.941 0.4893
     KIR      CAN    337.503 0.4900
     TLS      HKG    794.085 0.4901
     KIR      USA    403.291 0.4912
     ABW      DNK    920.766 0.4912
```