### `raw_data/disease_data/`

Place **non-WPP** raw disease datasets here (do not modify after adding).

Examples:
- AIDSVu exports (HIV prevalence/incidence)
- CDC WONDER exports (mortality tables)
- BRFSS extracts (risk factor prevalence)
- Local health department datasets (NYC DOHMH)

Cleaning/ETL scripts should live in `data_prep/` and write standardized outputs to `mighti/data/`
with `{region}_...` filenames.

