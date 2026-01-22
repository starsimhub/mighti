### `raw_data/` (do not modify once added)

This folder contains **immutable raw inputs** used to build MIGHTI-ready region files in `mighti/data/`.

Policy:
- Add new raw datasets here
- Do **not** edit/overwrite raw files after they are added (treat as provenance artifacts)

Structure:
- `raw_data/wpp_data/`: UN/WPP and related global raw inputs (CSVs)
- `raw_data/disease_data/`: non-WPP disease inputs (e.g., AIDSVu, CDC WONDER exports, BRFSS extracts)
- `raw_data/us_data/`: US/NYC-specific drop-in raw inputs for `us_data/region_data_builder.py`

All cleaned/derived outputs should be written to `mighti/data/` with a `{region}_...` prefix.

