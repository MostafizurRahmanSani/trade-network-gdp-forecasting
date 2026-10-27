# Look-ahead in the imputation of node features

Raw WDI series, the 182 economies of the main sample. Window of origin t = input years t-5..t. 'future-dependent' = the filled value depends on an observation after the origin t.

## infl

Raw missing cells 1994-2024: 531; filled by the interpolation: 117 (interior 10, leading/back-filled 75, trailing/forward-filled 32). Filled cells lying in gaps longer than three years (pandas fills up to three cells from each end of such a gap): 81.

| split | window cells (country x year) | filled cells in windows | of which future-dependent | samples (country x origin) | samples with >=1 future-dependent cell | share of samples |
|---|---|---|---|---|---|---|
| validation (origins 2014-2016) | 3276 | 18 | 3 | 546 | 2 | 0.4% |
| test (origins 2017-2023) | 7644 | 125 | 12 | 1274 | 6 | 0.5% |

Size of the issue: for interior filled cells, mean absolute difference between the interpolated value and a causal forward fill (last observed value): 3.21 (percentage points of inflation); n = 10.

Economies with future-dependent filled cells in any test window: ARG, BRB, SUR

## emp

Raw missing cells 1994-2024: 379; filled by the interpolation: 7 (interior 0, leading/back-filled 0, trailing/forward-filled 7). Filled cells lying in gaps longer than three years (pandas fills up to three cells from each end of such a gap): 0.

| split | window cells (country x year) | filled cells in windows | of which future-dependent | samples (country x origin) | samples with >=1 future-dependent cell | share of samples |
|---|---|---|---|---|---|---|
| validation (origins 2014-2016) | 3276 | 0 | 0 | 546 | 0 | 0.0% |
| test (origins 2017-2023) | 7644 | 5 | 0 | 1274 | 0 | 0.0% |
Economies with future-dependent filled cells in any test window: 

## Inflation winsorisation

Thresholds (annual %): full panel 1994-2024 [-1.96, 119.06] (used); training input years 1994-2013 [-2.23, 153.90].
After the training-period clipping applied in panel_data.build, the final inflation feature differs between the two versions in:

- validation (origins 2014-2016): 17 of 1456 country-year cells of the input years 2009-2016
- test (origins 2017-2023): 15 of 2184 country-year cells of the input years 2012-2023
- all years 1996-2024: 56 cells; largest absolute difference in the transformed feature: 0.0026
