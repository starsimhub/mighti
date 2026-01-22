### US/NYC data prep (drop-in raw → MIGHTI-ready)

This folder mirrors the role of the raw WPP inputs (now under `raw_data/wpp_data/`), but for US/NYC sources.

The goal is to generate **MIGHTI-ready** files in `mighti/data/` without overwriting anything unless asked.

## What MIGHTI expects per region

To run `mighti_main.py` or `mighti_life_expectancy.py`, the repo expects:
- `mighti/data/{region}_mx.csv` (wide mx table)
- `mighti/data/{region}_age_distribution.csv` (wide population age distribution)
- `mighti/data/{region}_prevalence.csv` (tidy prevalence grid by Age/Year with `<condition>_{male,female}` columns)
- `mighti/data/{region}_asfr.csv` (fertility table)

Then `prepare_data_for_year.py` derives:
- `mighti/data/{region}_mortality_rates.csv`
- `mighti/data/{region}_age_distribution_{inityear}.csv`

## Drop-in raw inputs (optional, recommended for NYC)

Place raw CSVs under:
`raw_data/us_data/{region}/`

Supported filenames:
- `mx_long.csv` with columns: `Age, Sex, Time, mx`
- `age_distribution_long.csv` with columns: `age, sex, year, value` (or `pop`)
- `hiv_prevalence_by_age_sex.csv` with columns: `Age, Year, HIV_male, HIV_female`

## Run the builder

From repo root:

```bash
python -c "from data_prep.us_data.region_data_builder import ensure_region_data; ensure_region_data(region='nyc', start_year=2007, end_year=2030, overwrite=False)"
```

If raw inputs are missing, the builder will create **placeholders** (so scripts run) and log warnings.

