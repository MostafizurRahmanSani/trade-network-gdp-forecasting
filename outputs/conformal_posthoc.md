# Conformal post-hoc analyses on saved CP-protocol forecasts (no refitting)

CP protocol: training targets <= 2012, early stopping 2013-2014, calibration 2015-2017, test 2018-2024. Bootstrap: 5,000 resamples of the 7 test years (90% percentile interval).

## A. Point accuracy of the CP-protocol forecasts

| Model | early-stopping MAE 2013-14 | calibration MAE 2015-17 | test MAE 2018-24 |
|---|---|---|---|
| STGNN | 0.02022 | 0.01984 | 0.03531 |
| LSTM | 0.01981 | 0.02000 | 0.03659 |
| GRU | 0.01986 | 0.01975 | 0.03651 |
| Rolling ARIMA | nan | 0.02189 | 0.03900 |
| No-change (naive) | nan | 0.03878 | 0.04816 |
| RW with drift (naive) | nan | 0.02469 | 0.03866 |

## B. Nominal 90%

| Model | method | coverage [90% CI] | cov. excl. 2020-21 | cov. 2020 | cov. 2021 | width | Winkler [90% CI] | WS excl. 2020-21 |
|---|---|---|---|---|---|---|---|---|
| STGNN | split CP | 0.788 [0.635, 0.911] | 0.911 | 0.269 | 0.692 | 0.097 | 0.358 [0.204, 0.587] | 0.213 |
|  | split coverage by volatility tercile (low/medium/high) | 0.862 / 0.783 / 0.719 | | | | | | |
| STGNN | Mondrian split CP (volatility terciles) | 0.747 [0.598, 0.874] | 0.873 | 0.264 | 0.604 | 0.097 | 0.367 [0.203, 0.605] | 0.208 |
|  | Mondrian coverage by volatility tercile (low/medium/high) | 0.616 / 0.757 / 0.869 | | | | | | |
| STGNN | expanding-window CP (no alpha update) | 0.835 [0.644, 0.952] | 0.952 | 0.231 | 0.857 | 0.129 | 0.367 [0.206, 0.625] | 0.223 |
| STGNN | ACI gamma 0.05 (as in paper) | 0.844 [0.637, 0.958] | 0.955 | 0.209 | 0.923 | 0.147 | 0.377 [0.212, 0.636] | 0.234 |
| STGNN | ACI gamma 0.01 | 0.837 [0.643, 0.951] | 0.951 | 0.231 | 0.874 | 0.130 | 0.367 [0.206, 0.626] | 0.224 |
| STGNN | ACI gamma 0.10 | 0.851 [0.641, 0.965] | 0.956 | 0.209 | 0.967 | 0.172 | 0.394 [0.225, 0.649] | 0.249 |
| STGNN | ACI gamma 0.05, corrected quantile | 0.847 [0.638, 0.961] | 0.956 | 0.209 | 0.940 | 0.148 | 0.377 [0.212, 0.635] | 0.234 |
| LSTM | split CP | 0.769 [0.615, 0.903] | 0.905 | 0.258 | 0.599 | 0.096 | 0.372 [0.208, 0.585] | 0.214 |
|  | split coverage by volatility tercile (low/medium/high) | 0.843 / 0.752 / 0.712 | | | | | | |
| LSTM | Mondrian split CP (volatility terciles) | 0.734 [0.579, 0.871] | 0.873 | 0.258 | 0.516 | 0.097 | 0.383 [0.208, 0.608] | 0.209 |
|  | Mondrian coverage by volatility tercile (low/medium/high) | 0.607 / 0.736 / 0.859 | | | | | | |
| LSTM | expanding-window CP (no alpha update) | 0.825 [0.641, 0.947] | 0.948 | 0.225 | 0.808 | 0.130 | 0.374 [0.211, 0.620] | 0.223 |
| LSTM | ACI gamma 0.05 (as in paper) | 0.840 [0.638, 0.958] | 0.957 | 0.209 | 0.885 | 0.150 | 0.381 [0.215, 0.633] | 0.234 |
| LSTM | ACI gamma 0.01 | 0.828 [0.642, 0.951] | 0.952 | 0.225 | 0.813 | 0.133 | 0.375 [0.212, 0.623] | 0.224 |
| LSTM | ACI gamma 0.10 | 0.849 [0.638, 0.964] | 0.957 | 0.203 | 0.951 | 0.174 | 0.394 [0.225, 0.643] | 0.249 |
| LSTM | ACI gamma 0.05, corrected quantile | 0.841 [0.643, 0.958] | 0.957 | 0.220 | 0.885 | 0.151 | 0.380 [0.216, 0.628] | 0.235 |
| GRU | split CP | 0.776 [0.621, 0.909] | 0.912 | 0.264 | 0.604 | 0.096 | 0.369 [0.206, 0.580] | 0.211 |
|  | split coverage by volatility tercile (low/medium/high) | 0.845 / 0.757 / 0.724 | | | | | | |
| GRU | Mondrian split CP (volatility terciles) | 0.729 [0.575, 0.867] | 0.869 | 0.253 | 0.505 | 0.095 | 0.383 [0.206, 0.610] | 0.207 |
|  | Mondrian coverage by volatility tercile (low/medium/high) | 0.602 / 0.729 / 0.857 | | | | | | |
| GRU | expanding-window CP (no alpha update) | 0.828 [0.640, 0.954] | 0.955 | 0.214 | 0.808 | 0.130 | 0.374 [0.209, 0.622] | 0.220 |
| GRU | ACI gamma 0.05 (as in paper) | 0.842 [0.641, 0.959] | 0.959 | 0.214 | 0.885 | 0.149 | 0.379 [0.212, 0.631] | 0.231 |
| GRU | ACI gamma 0.01 | 0.827 [0.639, 0.953] | 0.954 | 0.214 | 0.808 | 0.132 | 0.375 [0.210, 0.626] | 0.221 |
| GRU | ACI gamma 0.10 | 0.851 [0.643, 0.965] | 0.958 | 0.214 | 0.951 | 0.173 | 0.391 [0.222, 0.639] | 0.246 |
| GRU | ACI gamma 0.05, corrected quantile | 0.843 [0.641, 0.961] | 0.960 | 0.214 | 0.885 | 0.150 | 0.379 [0.213, 0.631] | 0.232 |
| Rolling ARIMA | split CP | 0.790 [0.650, 0.917] | 0.919 | 0.346 | 0.588 | 0.109 | 0.386 [0.218, 0.585] | 0.218 |
|  | split coverage by volatility tercile (low/medium/high) | 0.831 / 0.800 / 0.738 | | | | | | |
| Rolling ARIMA | Mondrian split CP (volatility terciles) | 0.743 [0.597, 0.875] | 0.878 | 0.297 | 0.516 | 0.103 | 0.404 [0.215, 0.623] | 0.211 |
|  | Mondrian coverage by volatility tercile (low/medium/high) | 0.630 / 0.755 / 0.845 | | | | | | |
| Rolling ARIMA | expanding-window CP (no alpha update) | 0.834 [0.669, 0.952] | 0.953 | 0.286 | 0.786 | 0.144 | 0.387 [0.226, 0.596] | 0.230 |
| Rolling ARIMA | ACI gamma 0.05 (as in paper) | 0.840 [0.666, 0.953] | 0.954 | 0.280 | 0.830 | 0.162 | 0.395 [0.234, 0.610] | 0.242 |
| Rolling ARIMA | ACI gamma 0.01 | 0.833 [0.665, 0.950] | 0.951 | 0.280 | 0.797 | 0.146 | 0.389 [0.227, 0.599] | 0.232 |
| Rolling ARIMA | ACI gamma 0.10 | 0.852 [0.665, 0.959] | 0.957 | 0.269 | 0.907 | 0.186 | 0.403 [0.242, 0.621] | 0.256 |
| Rolling ARIMA | ACI gamma 0.05, corrected quantile | 0.841 [0.667, 0.954] | 0.955 | 0.280 | 0.835 | 0.164 | 0.395 [0.234, 0.610] | 0.243 |
| No-change (naive) | split CP | 0.829 [0.761, 0.892] | 0.887 | 0.665 | 0.703 | 0.137 | 0.328 [0.237, 0.434] | 0.247 |
|  | split coverage by volatility tercile (low/medium/high) | 0.892 / 0.819 / 0.775 | | | | | | |
| No-change (naive) | Mondrian split CP (volatility terciles) | 0.849 [0.784, 0.911] | 0.905 | 0.676 | 0.742 | 0.146 | 0.324 [0.238, 0.425] | 0.247 |
|  | Mondrian coverage by volatility tercile (low/medium/high) | 0.890 / 0.781 / 0.876 | | | | | | |
| No-change (naive) | expanding-window CP (no alpha update) | 0.854 [0.784, 0.917] | 0.919 | 0.659 | 0.725 | 0.150 | 0.325 [0.237, 0.430] | 0.246 |
| No-change (naive) | ACI gamma 0.05 (as in paper) | 0.865 [0.794, 0.929] | 0.931 | 0.659 | 0.742 | 0.155 | 0.325 [0.238, 0.432] | 0.248 |
| No-change (naive) | ACI gamma 0.01 | 0.855 [0.785, 0.918] | 0.920 | 0.659 | 0.725 | 0.151 | 0.325 [0.237, 0.431] | 0.246 |
| No-change (naive) | ACI gamma 0.10 | 0.875 [0.808, 0.934] | 0.935 | 0.659 | 0.791 | 0.161 | 0.325 [0.239, 0.433] | 0.250 |
| No-change (naive) | ACI gamma 0.05, corrected quantile | 0.867 [0.797, 0.930] | 0.932 | 0.659 | 0.747 | 0.156 | 0.325 [0.238, 0.431] | 0.248 |
| RW with drift (naive) | split CP | 0.819 [0.680, 0.919] | 0.919 | 0.357 | 0.786 | 0.118 | 0.365 [0.224, 0.589] | 0.235 |
|  | split coverage by volatility tercile (low/medium/high) | 0.895 / 0.831 / 0.733 | | | | | | |
| RW with drift (naive) | Mondrian split CP (volatility terciles) | 0.771 [0.624, 0.887] | 0.886 | 0.280 | 0.687 | 0.114 | 0.369 [0.213, 0.611] | 0.220 |
|  | Mondrian coverage by volatility tercile (low/medium/high) | 0.663 / 0.779 / 0.871 | | | | | | |
| RW with drift (naive) | expanding-window CP (no alpha update) | 0.845 [0.662, 0.947] | 0.943 | 0.280 | 0.918 | 0.147 | 0.379 [0.227, 0.626] | 0.244 |
| RW with drift (naive) | ACI gamma 0.05 (as in paper) | 0.853 [0.664, 0.957] | 0.949 | 0.275 | 0.951 | 0.161 | 0.388 [0.233, 0.636] | 0.252 |
| RW with drift (naive) | ACI gamma 0.01 | 0.845 [0.662, 0.948] | 0.944 | 0.280 | 0.918 | 0.148 | 0.380 [0.227, 0.630] | 0.244 |
| RW with drift (naive) | ACI gamma 0.10 | 0.856 [0.664, 0.964] | 0.954 | 0.264 | 0.962 | 0.185 | 0.404 [0.247, 0.647] | 0.267 |
| RW with drift (naive) | ACI gamma 0.05, corrected quantile | 0.854 [0.666, 0.957] | 0.949 | 0.280 | 0.951 | 0.164 | 0.388 [0.235, 0.634] | 0.253 |

## B. Nominal 95%

| Model | method | coverage [90% CI] | cov. excl. 2020-21 | cov. 2020 | cov. 2021 | width | Winkler [90% CI] | WS excl. 2020-21 |
|---|---|---|---|---|---|---|---|---|
| STGNN | split CP | 0.884 [0.779, 0.958] | 0.959 | 0.538 | 0.852 | 0.146 | 0.512 [0.297, 0.842] | 0.320 |
| STGNN | Mondrian split CP (volatility terciles) | 0.811 [0.680, 0.921] | 0.921 | 0.368 | 0.703 | 0.136 | 0.554 [0.291, 0.948] | 0.302 |
| STGNN | expanding-window CP (no alpha update) | 0.891 [0.743, 0.970] | 0.969 | 0.440 | 0.951 | 0.190 | 0.538 [0.296, 0.929] | 0.331 |
| STGNN | ACI gamma 0.05 (as in paper) | 0.897 [0.739, 0.980] | 0.976 | 0.418 | 0.984 | 0.237 | 0.563 [0.313, 0.952] | 0.352 |
| STGNN | ACI gamma 0.01 | 0.889 [0.739, 0.970] | 0.969 | 0.429 | 0.951 | 0.192 | 0.539 [0.294, 0.933] | 0.331 |
| STGNN | ACI gamma 0.10 | 0.903 [0.744, 0.989] | 0.981 | 0.418 | 1.000 | 0.407 | 0.697 [0.418, 1.025] | 0.432 |
| STGNN | ACI gamma 0.05, corrected quantile | 0.899 [0.744, 0.980] | 0.976 | 0.429 | 0.984 | 0.241 | 0.563 [0.317, 0.945] | 0.355 |
| LSTM | split CP | 0.859 [0.746, 0.952] | 0.954 | 0.478 | 0.764 | 0.137 | 0.538 [0.304, 0.863] | 0.317 |
| LSTM | Mondrian split CP (volatility terciles) | 0.794 [0.658, 0.917] | 0.919 | 0.352 | 0.615 | 0.128 | 0.584 [0.300, 0.963] | 0.303 |
| LSTM | expanding-window CP (no alpha update) | 0.889 [0.744, 0.972] | 0.971 | 0.440 | 0.929 | 0.191 | 0.537 [0.295, 0.917] | 0.327 |
| LSTM | ACI gamma 0.05 (as in paper) | 0.892 [0.724, 0.980] | 0.977 | 0.385 | 0.978 | 0.245 | 0.569 [0.314, 0.962] | 0.353 |
| LSTM | ACI gamma 0.01 | 0.884 [0.725, 0.973] | 0.973 | 0.396 | 0.929 | 0.192 | 0.546 [0.293, 0.941] | 0.327 |
| LSTM | ACI gamma 0.10 | 0.897 [0.723, 0.989] | 0.982 | 0.368 | 1.000 | 0.409 | 0.701 [0.418, 1.038] | 0.433 |
| LSTM | ACI gamma 0.05, corrected quantile | 0.894 [0.729, 0.980] | 0.977 | 0.396 | 0.978 | 0.247 | 0.568 [0.316, 0.956] | 0.354 |
| GRU | split CP | 0.859 [0.749, 0.954] | 0.955 | 0.484 | 0.758 | 0.138 | 0.535 [0.299, 0.857] | 0.313 |
| GRU | Mondrian split CP (volatility terciles) | 0.796 [0.657, 0.920] | 0.923 | 0.357 | 0.599 | 0.129 | 0.582 [0.297, 0.959] | 0.299 |
| GRU | expanding-window CP (no alpha update) | 0.883 [0.725, 0.973] | 0.973 | 0.396 | 0.923 | 0.189 | 0.542 [0.293, 0.938] | 0.323 |
| GRU | ACI gamma 0.05 (as in paper) | 0.893 [0.725, 0.981] | 0.978 | 0.385 | 0.978 | 0.242 | 0.567 [0.309, 0.963] | 0.347 |
| GRU | ACI gamma 0.01 | 0.882 [0.721, 0.973] | 0.973 | 0.385 | 0.929 | 0.192 | 0.546 [0.290, 0.948] | 0.323 |
| GRU | ACI gamma 0.10 | 0.899 [0.730, 0.989] | 0.981 | 0.385 | 1.000 | 0.408 | 0.699 [0.413, 1.037] | 0.430 |
| GRU | ACI gamma 0.05, corrected quantile | 0.893 [0.725, 0.981] | 0.978 | 0.385 | 0.978 | 0.248 | 0.571 [0.314, 0.964] | 0.352 |
| Rolling ARIMA | split CP | 0.850 [0.746, 0.945] | 0.946 | 0.505 | 0.714 | 0.144 | 0.575 [0.315, 0.880] | 0.317 |
| Rolling ARIMA | Mondrian split CP (volatility terciles) | 0.809 [0.679, 0.929] | 0.930 | 0.390 | 0.626 | 0.151 | 0.610 [0.309, 0.965] | 0.305 |
| Rolling ARIMA | expanding-window CP (no alpha update) | 0.889 [0.765, 0.965] | 0.966 | 0.495 | 0.901 | 0.204 | 0.549 [0.312, 0.873] | 0.323 |
| Rolling ARIMA | ACI gamma 0.05 (as in paper) | 0.896 [0.755, 0.975] | 0.973 | 0.462 | 0.951 | 0.250 | 0.569 [0.329, 0.912] | 0.347 |
| Rolling ARIMA | ACI gamma 0.01 | 0.885 [0.750, 0.965] | 0.965 | 0.462 | 0.907 | 0.207 | 0.557 [0.314, 0.896] | 0.325 |
| Rolling ARIMA | ACI gamma 0.10 | 0.906 [0.761, 0.987] | 0.977 | 0.462 | 0.995 | 0.376 | 0.647 [0.394, 0.956] | 0.421 |
| Rolling ARIMA | ACI gamma 0.05, corrected quantile | 0.897 [0.756, 0.976] | 0.974 | 0.462 | 0.951 | 0.257 | 0.569 [0.331, 0.907] | 0.351 |
| No-change (naive) | split CP | 0.896 [0.840, 0.948] | 0.945 | 0.742 | 0.808 | 0.170 | 0.464 [0.320, 0.638] | 0.339 |
| No-change (naive) | Mondrian split CP (volatility terciles) | 0.889 [0.833, 0.941] | 0.937 | 0.747 | 0.786 | 0.175 | 0.454 [0.315, 0.623] | 0.332 |
| No-change (naive) | expanding-window CP (no alpha update) | 0.916 [0.856, 0.968] | 0.968 | 0.720 | 0.852 | 0.192 | 0.463 [0.315, 0.660] | 0.337 |
| No-change (naive) | ACI gamma 0.05 (as in paper) | 0.922 [0.858, 0.972] | 0.973 | 0.709 | 0.879 | 0.203 | 0.465 [0.316, 0.663] | 0.341 |
| No-change (naive) | ACI gamma 0.01 | 0.915 [0.855, 0.968] | 0.968 | 0.714 | 0.852 | 0.192 | 0.465 [0.315, 0.664] | 0.337 |
| No-change (naive) | ACI gamma 0.10 | 0.929 [0.861, 0.975] | 0.975 | 0.709 | 0.923 | 0.221 | 0.466 [0.315, 0.666] | 0.348 |
| No-change (naive) | ACI gamma 0.05, corrected quantile | 0.922 [0.860, 0.972] | 0.973 | 0.714 | 0.879 | 0.206 | 0.464 [0.316, 0.662] | 0.342 |
| RW with drift (naive) | split CP | 0.892 [0.786, 0.961] | 0.956 | 0.555 | 0.912 | 0.171 | 0.515 [0.316, 0.831] | 0.342 |
| RW with drift (naive) | Mondrian split CP (volatility terciles) | 0.840 [0.712, 0.939] | 0.938 | 0.412 | 0.775 | 0.178 | 0.542 [0.302, 0.917] | 0.317 |
| RW with drift (naive) | expanding-window CP (no alpha update) | 0.902 [0.772, 0.973] | 0.970 | 0.505 | 0.956 | 0.215 | 0.535 [0.321, 0.877] | 0.353 |
| RW with drift (naive) | ACI gamma 0.05 (as in paper) | 0.905 [0.768, 0.980] | 0.975 | 0.484 | 0.978 | 0.263 | 0.570 [0.343, 0.924] | 0.377 |
| RW with drift (naive) | ACI gamma 0.01 | 0.901 [0.768, 0.974] | 0.971 | 0.495 | 0.956 | 0.215 | 0.543 [0.322, 0.897] | 0.354 |
| RW with drift (naive) | ACI gamma 0.10 | 0.911 [0.772, 0.989] | 0.978 | 0.484 | 1.000 | 0.353 | 0.632 [0.396, 0.956] | 0.427 |
| RW with drift (naive) | ACI gamma 0.05, corrected quantile | 0.908 [0.773, 0.982] | 0.976 | 0.495 | 0.984 | 0.266 | 0.567 [0.344, 0.911] | 0.378 |

## C. Year-clustered DM test on the Winkler loss (split CP, 90%; negative = STGNN better; HLN, t(6))

| Comparison | mean diff | p | years STGNN better |
|---|---|---|---|
| STGNN vs LSTM | -0.0141 | 0.427 | 2 |
| STGNN vs GRU | -0.0115 | 0.542 | 2 |
| STGNN vs Rolling ARIMA | -0.0289 | 0.508 | 4 |
| STGNN vs No-change (naive) | +0.0293 | 0.694 | 6 |
| STGNN vs RW with drift (naive) | -0.0075 | 0.522 | 5 |
| STGNN: ACI vs split CP | +0.0198 | 0.409 | 3 (ACI better) |

## D. ACI path for the STGNN at 90% (alpha used in each test year, q, coverage)

| year | alpha_t | q | coverage |
|---|---|---|---|
| 2018 | 0.1000 | 0.0475 | 0.956 |
| 2019 | 0.1028 | 0.0433 | 0.929 |
| 2020 | 0.1042 | 0.0407 | 0.209 |
| 2021 | 0.0697 | 0.1003 | 0.923 |
| 2022 | 0.0708 | 0.1003 | 0.940 |
| 2023 | 0.0728 | 0.0960 | 0.973 |
| 2024 | 0.0764 | 0.0880 | 0.978 |
