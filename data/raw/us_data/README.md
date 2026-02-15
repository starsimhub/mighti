### `raw_data/us_data/`

Drop-in raw inputs for US/NYC runs.

For a region like `nyc`, put files under:
`raw_data/us_data/nyc/`

Supported filenames (optional, but recommended):
- `mx_long.csv` with columns: `Age, Sex, Time, mx`
- `age_distribution_long.csv` with columns: `age, sex, year, value` (or `pop`)
- `hiv_prevalence_by_age_sex.csv` with columns: `Age, Year, HIV_male, HIV_female`

These will be consumed by `us_data/region_data_builder.py` to generate (if missing):
`mighti/data/nyc_mx.csv`, `mighti/data/nyc_age_distribution.csv`, `mighti/data/nyc_prevalence.csv`, etc.

Note: the builder code lives in `data_prep/us_data/region_data_builder.py`.
