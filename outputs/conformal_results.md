# Conformal prediction (calibration targets 2015-17 separate from early stopping 2013-14; test 2018-24)

## Nominal coverage 90%

```
               model                 method  coverage  width  winkler     q   mae  cov2018  cov2019  cov2020  cov2021  cov2022  cov2023  cov2024
 STGNN (5-seed mean)               split-CP     0.786  0.096    0.359 0.048 0.035    0.956    0.934    0.253    0.698    0.808    0.918    0.934
 STGNN (5-seed mean)          normalised-CP     0.759  0.105    0.344 1.229 0.035    0.945    0.901    0.209    0.621    0.802    0.890    0.945
 STGNN (5-seed mean)                    ACI     0.842  0.148    0.378 0.074 0.035    0.956    0.918    0.209    0.923    0.940    0.973    0.978
 STGNN (5-seed mean)             rolling-CP     0.846  0.158    0.387 0.079 0.035    0.956    0.923    0.192    0.945    0.956    0.973    0.978
  LSTM (5-seed mean)               split-CP     0.769  0.096    0.372 0.048 0.037    0.956    0.934    0.258    0.599    0.786    0.896    0.956
  LSTM (5-seed mean)          normalised-CP     0.741  0.104    0.357 1.220 0.037    0.940    0.896    0.214    0.527    0.775    0.896    0.940
  LSTM (5-seed mean)                    ACI     0.840  0.150    0.381 0.075 0.037    0.956    0.929    0.209    0.885    0.945    0.973    0.984
  LSTM (5-seed mean)             rolling-CP     0.841  0.163    0.390 0.081 0.037    0.956    0.929    0.192    0.912    0.951    0.973    0.978
   GRU (5-seed mean)               split-CP     0.776  0.096    0.369 0.048 0.037    0.967    0.940    0.264    0.604    0.791    0.901    0.962
   GRU (5-seed mean)          normalised-CP     0.742  0.104    0.355 1.225 0.037    0.934    0.896    0.214    0.533    0.775    0.901    0.940
   GRU (5-seed mean)                    ACI     0.842  0.149    0.379 0.074 0.037    0.962    0.934    0.214    0.885    0.945    0.973    0.984
   GRU (5-seed mean)             rolling-CP     0.846  0.163    0.387 0.081 0.037    0.967    0.934    0.209    0.907    0.956    0.973    0.978
                 GBM               split-CP     0.795  0.100    0.360 0.050 0.036    0.962    0.934    0.286    0.703    0.841    0.901    0.940
                 GBM          normalised-CP     0.754  0.103    0.353 1.211 0.036    0.940    0.890    0.187    0.632    0.802    0.879    0.945
                 GBM                    ACI     0.842  0.149    0.384 0.075 0.036    0.956    0.912    0.187    0.945    0.945    0.973    0.978
                 GBM             rolling-CP     0.845  0.161    0.390 0.080 0.036    0.962    0.918    0.187    0.951    0.956    0.973    0.973
        RandomForest               split-CP     0.795  0.100    0.355 0.050 0.036    0.962    0.934    0.313    0.659    0.841    0.901    0.956
        RandomForest          normalised-CP     0.755  0.103    0.347 1.204 0.036    0.934    0.901    0.220    0.604    0.797    0.879    0.951
        RandomForest                    ACI     0.841  0.145    0.375 0.073 0.036    0.962    0.907    0.220    0.912    0.934    0.973    0.978
        RandomForest             rolling-CP     0.849  0.157    0.380 0.078 0.036    0.962    0.918    0.220    0.940    0.951    0.973    0.978
        AR(1) pooled               split-CP     0.774  0.092    0.388 0.046 0.038    0.962    0.934    0.247    0.610    0.857    0.874    0.934
        AR(1) pooled          normalised-CP     0.774  0.124    0.351 1.451 0.038    0.940    0.885    0.253    0.615    0.868    0.907    0.951
        AR(1) pooled                    ACI     0.838  0.155    0.396 0.077 0.038    0.962    0.907    0.220    0.879    0.945    0.967    0.984
        AR(1) pooled             rolling-CP     0.845  0.166    0.404 0.083 0.038    0.962    0.907    0.187    0.929    0.962    0.984    0.984
       ARIMA rolling               split-CP     0.751  0.100    0.419 0.050 0.041    0.951    0.956    0.390    0.418    0.731    0.885    0.929
       ARIMA rolling          normalised-CP     0.735  0.108    0.402 1.267 0.041    0.934    0.923    0.280    0.412    0.725    0.923    0.951
       ARIMA rolling                    ACI     0.829  0.162    0.410 0.081 0.041    0.951    0.923    0.335    0.703    0.934    0.973    0.984
       ARIMA rolling             rolling-CP     0.839  0.183    0.415 0.092 0.041    0.951    0.929    0.313    0.775    0.940    0.984    0.984
ARIMA rolling, drift               split-CP     0.790  0.109    0.386 0.054 0.039    0.962    0.940    0.346    0.588    0.830    0.912    0.951
ARIMA rolling, drift          normalised-CP     0.761  0.113    0.376 1.331 0.039    0.940    0.896    0.258    0.527    0.786    0.940    0.978
ARIMA rolling, drift                    ACI     0.840  0.162    0.395 0.081 0.039    0.956    0.934    0.280    0.830    0.934    0.967    0.978
ARIMA rolling, drift             rolling-CP     0.845  0.177    0.402 0.089 0.039    0.962    0.934    0.269    0.846    0.945    0.978    0.978
        GBM-quantile        native quantile     0.809  0.100    0.332 0.050 0.035    0.962    0.956    0.264    0.764    0.841    0.934    0.945
     Linear-quantile        native quantile     0.811  0.116    0.355 0.058 0.037    0.984    0.978    0.357    0.610    0.863    0.912    0.973
 STGNN deep ensemble Gaussian (seed spread)     0.106  0.006    0.652 0.003 0.035    0.192    0.159    0.000    0.044    0.082    0.132    0.132
```

Coverage by country group (split-CP):

```
               model  cov[size:small]  cov[size:medium]  cov[size:large]  cov[volatility:low]  cov[volatility:medium]  cov[volatility:high]  cov[income(GDPpc proxy):low]  cov[income(GDPpc proxy):middle]  cov[income(GDPpc proxy):high]
 STGNN (5-seed mean)            0.775             0.764            0.817                0.862                   0.781                 0.714                         0.834                            0.752                          0.770
  LSTM (5-seed mean)            0.761             0.762            0.785                0.843                   0.752                 0.712                         0.829                            0.743                          0.735
   GRU (5-seed mean)            0.763             0.767            0.796                0.845                   0.757                 0.724                         0.834                            0.745                          0.747
                 GBM            0.782             0.783            0.820                0.869                   0.793                 0.724                         0.843                            0.745                          0.796
        RandomForest            0.782             0.790            0.813                0.855                   0.795                 0.735                         0.850                            0.748                          0.787
        AR(1) pooled            0.766             0.771            0.785                0.841                   0.764                 0.717                         0.829                            0.740                          0.752
       ARIMA rolling            0.724             0.750            0.780                0.806                   0.757                 0.691                         0.836                            0.698                          0.719
ARIMA rolling, drift            0.778             0.776            0.815                0.831                   0.800                 0.738                         0.862                            0.745                          0.761
```

## Nominal coverage 95%

```
               model                 method  coverage  width  winkler     q   mae  cov2018  cov2019  cov2020  cov2021  cov2022  cov2023  cov2024
 STGNN (5-seed mean)               split-CP     0.882  0.147    0.512 0.073 0.035    0.978    0.973    0.533    0.846    0.912    0.956    0.978
 STGNN (5-seed mean)          normalised-CP     0.851  0.155    0.477 1.817 0.035    0.973    0.978    0.352    0.780    0.923    0.973    0.978
 STGNN (5-seed mean)                    ACI     0.897  0.236    0.565 0.118 0.035    0.978    0.956    0.423    0.984    0.978    0.978    0.984
 STGNN (5-seed mean)             rolling-CP     0.900  0.231    0.553 0.116 0.035    0.978    0.962    0.440    0.984    0.978    0.978    0.978
  LSTM (5-seed mean)               split-CP     0.859  0.137    0.538 0.069 0.037    0.984    0.967    0.478    0.764    0.890    0.951    0.978
  LSTM (5-seed mean)          normalised-CP     0.840  0.155    0.490 1.818 0.037    0.973    0.973    0.357    0.698    0.918    0.978    0.984
  LSTM (5-seed mean)                    ACI     0.892  0.245    0.569 0.123 0.037    0.984    0.967    0.385    0.978    0.973    0.978    0.984
  LSTM (5-seed mean)             rolling-CP     0.889  0.233    0.565 0.117 0.037    0.984    0.967    0.368    0.973    0.973    0.978    0.984
   GRU (5-seed mean)               split-CP     0.859  0.138    0.535 0.069 0.037    0.984    0.973    0.484    0.758    0.890    0.951    0.978
   GRU (5-seed mean)          normalised-CP     0.837  0.153    0.491 1.794 0.037    0.973    0.973    0.363    0.676    0.912    0.978    0.984
   GRU (5-seed mean)                    ACI     0.893  0.242    0.567 0.121 0.037    0.984    0.967    0.385    0.978    0.973    0.978    0.989
   GRU (5-seed mean)             rolling-CP     0.887  0.234    0.568 0.117 0.037    0.984    0.967    0.352    0.973    0.973    0.978    0.984
                 GBM               split-CP     0.870  0.142    0.525 0.071 0.036    0.978    0.967    0.462    0.863    0.907    0.945    0.973
                 GBM          normalised-CP     0.842  0.153    0.488 1.798 0.036    0.967    0.967    0.346    0.786    0.907    0.951    0.973
                 GBM                    ACI     0.899  0.234    0.563 0.117 0.036    0.978    0.967    0.423    0.989    0.973    0.978    0.984
                 GBM             rolling-CP     0.899  0.234    0.557 0.117 0.036    0.978    0.967    0.429    0.984    0.978    0.978    0.978
        RandomForest               split-CP     0.871  0.138    0.520 0.069 0.036    0.978    0.967    0.495    0.819    0.912    0.951    0.978
        RandomForest          normalised-CP     0.834  0.144    0.492 1.685 0.036    0.973    0.978    0.330    0.747    0.896    0.945    0.967
        RandomForest                    ACI     0.895  0.237    0.556 0.119 0.036    0.978    0.956    0.412    0.984    0.973    0.978    0.984
        RandomForest             rolling-CP     0.898  0.232    0.542 0.116 0.036    0.978    0.956    0.434    0.984    0.978    0.978    0.978
        AR(1) pooled               split-CP     0.855  0.139    0.558 0.070 0.038    0.978    0.973    0.440    0.769    0.907    0.940    0.978
        AR(1) pooled          normalised-CP     0.838  0.158    0.507 1.856 0.038    0.962    0.973    0.374    0.720    0.912    0.951    0.973
        AR(1) pooled                    ACI     0.889  0.260    0.604 0.130 0.038    0.978    0.967    0.352    0.973    0.973    0.989    0.989
        AR(1) pooled             rolling-CP     0.882  0.242    0.600 0.121 0.038    0.978    0.967    0.319    0.962    0.973    0.989    0.989
       ARIMA rolling               split-CP     0.826  0.132    0.636 0.066 0.041    0.973    0.967    0.511    0.560    0.852    0.945    0.973
       ARIMA rolling          normalised-CP     0.810  0.155    0.567 1.816 0.041    0.978    0.967    0.385    0.516    0.868    0.973    0.984
       ARIMA rolling                    ACI     0.894  0.250    0.579 0.125 0.041    0.973    0.962    0.495    0.896    0.962    0.984    0.989
       ARIMA rolling             rolling-CP     0.893  0.259    0.591 0.130 0.041    0.973    0.967    0.473    0.896    0.973    0.984    0.989
ARIMA rolling, drift               split-CP     0.850  0.144    0.575 0.072 0.039    0.973    0.956    0.505    0.714    0.885    0.945    0.973
ARIMA rolling, drift          normalised-CP     0.834  0.163    0.523 1.915 0.039    0.967    0.973    0.368    0.659    0.907    0.978    0.989
ARIMA rolling, drift                    ACI     0.896  0.250    0.569 0.125 0.039    0.967    0.951    0.462    0.951    0.967    0.984    0.995
ARIMA rolling, drift             rolling-CP     0.901  0.257    0.562 0.128 0.039    0.973    0.951    0.495    0.951    0.967    0.984    0.989
        GBM-quantile        native quantile     0.872  0.132    0.496 0.066 0.036    0.984    0.973    0.429    0.896    0.901    0.951    0.973
     Linear-quantile        native quantile     0.882  0.161    0.504 0.080 0.038    0.984    0.984    0.516    0.824    0.918    0.967    0.984
 STGNN deep ensemble Gaussian (seed spread)     0.124  0.008    1.277 0.004 0.035    0.203    0.176    0.000    0.060    0.121    0.137    0.170
```

Coverage by country group (split-CP):

```
               model  cov[size:small]  cov[size:medium]  cov[size:large]  cov[volatility:low]  cov[volatility:medium]  cov[volatility:high]  cov[income(GDPpc proxy):low]  cov[income(GDPpc proxy):middle]  cov[income(GDPpc proxy):high]
 STGNN (5-seed mean)            0.857             0.879            0.911                0.925                   0.871                 0.850                         0.916                            0.848                          0.883
  LSTM (5-seed mean)            0.843             0.857            0.876                0.902                   0.860                 0.815                         0.904                            0.824                          0.848
   GRU (5-seed mean)            0.843             0.857            0.878                0.902                   0.860                 0.817                         0.902                            0.829                          0.848
                 GBM            0.850             0.864            0.897                0.918                   0.864                 0.829                         0.906                            0.833                          0.871
        RandomForest            0.845             0.874            0.895                0.918                   0.867                 0.829                         0.909                            0.838                          0.867
        AR(1) pooled            0.841             0.850            0.874                0.904                   0.850                 0.810                         0.902                            0.817                          0.845
       ARIMA rolling            0.806             0.829            0.843                0.859                   0.833                 0.785                         0.892                            0.788                          0.796
ARIMA rolling, drift            0.829             0.838            0.883                0.892                   0.855                 0.803                         0.904                            0.812                          0.834
```

## Winkler score by test year and by country group (split-CP, 90%)

```
               model  wink2018  wink2019  wink2020  wink2021  wink2022  wink2023  wink2024  wink[size:small]  wink[size:medium]  wink[size:large]  wink[volatility:low]  wink[volatility:medium]  wink[volatility:high]  wink[income(GDPpc proxy):low]  wink[income(GDPpc proxy):middle]  wink[income(GDPpc proxy):high]
 STGNN (5-seed mean)     0.130     0.177     1.106     0.338     0.323     0.255     0.183             0.440              0.413             0.224                 0.192                    0.348                  0.536                          0.296                             0.435                           0.346
  LSTM (5-seed mean)     0.125     0.171     1.093     0.440     0.349     0.253     0.171             0.450              0.423             0.243                 0.205                    0.363                  0.547                          0.293                             0.448                           0.375
   GRU (5-seed mean)     0.121     0.167     1.084     0.444     0.349     0.250     0.168             0.450              0.416             0.242                 0.205                    0.362                  0.540                          0.286                             0.447                           0.375
                 GBM     0.137     0.184     1.110     0.314     0.317     0.267     0.190             0.439              0.415             0.227                 0.193                    0.349                  0.539                          0.302                             0.435                           0.345
        RandomForest     0.134     0.182     1.063     0.339     0.323     0.261     0.183             0.435              0.409             0.221                 0.190                    0.342                  0.531                          0.293                             0.429                           0.343
        AR(1) pooled     0.122     0.165     1.195     0.486     0.321     0.258     0.167             0.459              0.445             0.261                 0.213                    0.371                  0.579                          0.283                             0.470                           0.412
       ARIMA rolling     0.145     0.150     0.975     0.845     0.389     0.259     0.173             0.496              0.480             0.283                 0.251                    0.396                  0.611                          0.296                             0.477                           0.486
ARIMA rolling, drift     0.144     0.159     1.031     0.584     0.359     0.260     0.168             0.465              0.442             0.254                 0.223                    0.370                  0.566                          0.288                             0.446                           0.426
```

## Seed variability of the interval metrics (each of the 5 seeds calibrated separately; mean and sd over seeds; all methods, both levels)

```
                          coverage           width         winkler        
                              mean     std    mean     std    mean     std
alpha method        model                                                 
0.05  ACI           GRU     0.8931  0.0015  0.2425  0.0021  0.5683  0.0033
                    LSTM    0.8932  0.0022  0.2436  0.0019  0.5691  0.0044
                    STGNN   0.8959  0.0018  0.2372  0.0023  0.5657  0.0030
      normalised-CP GRU     0.8295  0.0063  0.1476  0.0037  0.5005  0.0058
                    LSTM    0.8356  0.0047  0.1518  0.0027  0.4959  0.0050
                    STGNN   0.8490  0.0038  0.1528  0.0031  0.4803  0.0046
      rolling-CP    GRU     0.8881  0.0021  0.2335  0.0011  0.5701  0.0052
                    LSTM    0.8898  0.0016  0.2333  0.0014  0.5657  0.0020
                    STGNN   0.8986  0.0015  0.2306  0.0013  0.5544  0.0033
      split-CP      GRU     0.8576  0.0027  0.1365  0.0017  0.5375  0.0035
                    LSTM    0.8571  0.0044  0.1363  0.0017  0.5397  0.0035
                    STGNN   0.8774  0.0054  0.1421  0.0035  0.5184  0.0066
0.10  ACI           GRU     0.8410  0.0013  0.1490  0.0008  0.3791  0.0008
                    LSTM    0.8386  0.0012  0.1505  0.0007  0.3811  0.0011
                    STGNN   0.8441  0.0016  0.1474  0.0008  0.3777  0.0021
      normalised-CP GRU     0.7391  0.0030  0.1032  0.0009  0.3568  0.0011
                    LSTM    0.7413  0.0045  0.1050  0.0015  0.3562  0.0023
                    STGNN   0.7590  0.0068  0.1047  0.0024  0.3443  0.0027
      rolling-CP    GRU     0.8436  0.0014  0.1630  0.0007  0.3885  0.0017
                    LSTM    0.8418  0.0007  0.1630  0.0007  0.3910  0.0014
                    STGNN   0.8473  0.0016  0.1589  0.0013  0.3868  0.0021
      split-CP      GRU     0.7771  0.0029  0.0974  0.0009  0.3679  0.0015
                    LSTM    0.7706  0.0030  0.0962  0.0006  0.3712  0.0021
                    STGNN   0.7873  0.0060  0.0969  0.0020  0.3584  0.0013
```

Spearman correlation between MAE and Winkler score across models (split-CP, 90%): 0.95 (p=0.000); with constant-width intervals this alignment is partly mechanical.

Effective calibration size: 546 residuals come from 3 calendar years; residuals within a year share global shocks, so the effective number of independent calibration points is closer to 3 than to 546.

Split-CP validity requires exchangeability between calibration and test residuals, which the 2020-21 shock violates; adaptive and rolling variants are reported for this reason.