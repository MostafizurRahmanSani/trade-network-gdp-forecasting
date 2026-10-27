# Full trade network vs the top-k graphs used by the model (182 countries)

Directed density of the complete BACI network (all exporter-importer pairs with positive trade, all products summed) and the median share of an exporter's total exports captured by its top-k destinations.

```
 year  full_edges  full_density  cover_median_k5  cover_median_k12  cover_median_k25  edges_k5  edges_k12  edges_k25
 1996       17037         0.517            0.664             0.866             0.969       875       2100       4339
 2000       21434         0.651            0.659             0.849             0.954       910       2184       4549
 2005       23314         0.708            0.640             0.840             0.954       910       2184       4550
 2010       24781         0.752            0.615             0.816             0.937       910       2184       4550
 2015       25380         0.770            0.582             0.798             0.932       910       2184       4550
 2019       26036         0.790            0.622             0.812             0.936       910       2184       4550
 2023       25870         0.785            0.606             0.821             0.939       910       2184       4550
```

The complete network is dense (directed density 0.52 in 1996, 0.79 in 2019-2023); the model graph (k = 12) has density about 0.065 only because each exporter keeps its 12 largest destinations, which carry a median 83% of its exports.