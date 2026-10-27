# Cleaning report

- WDI rows: 266; kept 182 economies present in BACI with complete real GDP & population 1994-2024.
- Dropped (aggregates or incomplete GDP/pop): 84
- Obsolete BACI codes merged into the current code of the same economy: {58: 56, 280: 276, 736: 729}
- Inflation missing cells: 531 ; employment-ratio missing cells: 379
- infl: 414 cells left after <=3y interpolation -> imputed with the cross-country median of that year
- emp: 372 cells left after <=3y interpolation -> imputed with the cross-country median of that year
- Inflation winsorised to [-2.0, 119.1]% (1st/99th percentile) then signed-log1p transformed
- Node panel: 5278 country-years, 15 features, years 1996-2024
- Extreme trade-growth values clipped later at the 1st/99th percentile of the training period
- Edges: per year, each exporter keeps its top-12 destinations by total value (BACI, all products)
