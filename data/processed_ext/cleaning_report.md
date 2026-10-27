# Cleaning report

- WDI rows: 266; kept 189 economies present in BACI with complete real GDP & population 1995-2023 (extended-country sensitivity set).
- Dropped (aggregates or incomplete GDP/pop): 77
- Obsolete BACI codes merged into the current code of the same economy: {58: 56, 280: 276, 736: 729}
- Inflation missing cells: 616 ; employment-ratio missing cells: 473
- infl: 486 cells left after <=3y interpolation -> imputed with the cross-country median of that year
- emp: 465 cells left after <=3y interpolation -> imputed with the cross-country median of that year
- Inflation winsorised to [-1.9, 121.7]% (1st/99th percentile) then signed-log1p transformed
- Node panel: 5481 country-years, 15 features, years 1996-2024
- Extreme trade-growth values clipped later at the 1st/99th percentile of the training period
- Edges: per year, each exporter keeps its top-12 destinations by total value (BACI, all products)
