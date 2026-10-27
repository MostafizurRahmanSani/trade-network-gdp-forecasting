# Shock-propagation experiment (final STGNN configuration, 5 seeds per graph variant)

A -5 percentage-point shock (log growth points) is injected into the 2019 growth input of one economy; the table gives the change in the predicted 2020 growth. 'own' = the shocked economy itself (mean ± sd over seeds); the other columns are the mean ABSOLUTE change in the forecasts of (a) the economies that receive its exports (its 12 destinations in the real 2019 graph), (b) its own exporters (economies that list it as a destination), (c) all economies not linked to it. Only the growth input is shocked; derived partner-growth inputs are unchanged, so the response isolates message passing.

| graph used for training/inference | shocked economy | own response | receivers (mean abs.) | exporters to it (mean abs.) | unlinked (mean abs.) |
|---|---|---|---|---|---|
| real | CHN | -0.0213 ± 0.0032 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| real | DEU | -0.0012 ± 0.0008 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| real | USA | -0.0049 ± 0.0017 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| reversed | CHN | -0.0225 ± 0.0029 | 1.87e-07 | 1.80e-07 | 0.00e+00 |
| reversed | DEU | -0.0012 ± 0.0008 | 1.11e-07 | 1.02e-07 | 0.00e+00 |
| reversed | USA | -0.0059 ± 0.0008 | 1.36e-07 | 1.22e-07 | 0.00e+00 |
| shuffled | CHN | -0.0212 ± 0.0033 | 4.97e-10 | 6.39e-11 | 0.00e+00 |
| shuffled | DEU | -0.0010 ± 0.0008 | 2.48e-10 | 5.37e-11 | 8.51e-11 |
| shuffled | USA | -0.0048 ± 0.0016 | 0.00e+00 | 3.97e-11 | 0.00e+00 |

Reading: the shocked economy's own forecast moves by about 0.5 times the shock (China -0.024 for a -0.05 shock), but forecasts of its trade partners change by less than 1e-8, i.e. numerically zero, for the real, the shuffled and the reversed graph alike. The trained STGNN therefore does not transmit information along trade edges; the temporal GRU per economy does all the work. Models without any cross-country input (LSTM, GRU, SVR, GBM, ...) have exactly zero cross-country response by construction.