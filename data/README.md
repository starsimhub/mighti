### `data/processed`

This folder contains **MIGHTI-ready, cleaned/derived inputs** that the model and
example scripts read directly. Files should be treated as *generated artifacts*
from `data/raw/` inputs using scripts in `data_prep/`.

**Policy**
- Prefer regenerating these files via `data_prep/` rather than editing them by hand.
- All outputs should use a `{region}_...` prefix (e.g. `eswatini_parameters.csv`).

**Typical contents**
- `{region}_parameters.csv`
- `{region}_prevalence.csv` and `{region}_prevalence_hiv.csv`
- `{region}_age_distribution.csv` and `{region}_age_distribution_{year}.csv`
- `{region}_mx.csv` and `{region}_mortality_rates.csv`
- `{region}_asfr.csv`
- `{region}_intervention.csv`
- `rel_sus.csv`
- `sdoh.csv`

---

### `data/raw` (gitignored)

This folder contains **immutable raw inputs** used to build the region files in
`data/processed/`.

Note: `data/raw/` is typically **gitignored**, so you may not see it in a fresh
clone unless you have the raw datasets locally.

**Policy**
- Add new raw datasets here
- Do **not** edit/overwrite raw files after they are added (treat as provenance artifacts)

**Structure (convention)**
- `data/raw/wpp_data/`: UN/WPP and related global raw inputs (CSVs)
- `data/raw/disease_data/`: disease inputs (typically downloaded from IHME GBD; can also include other sources)

All cleaned/derived outputs should be written to `data/processed/` with a
`{region}_...` prefix.

---

### Where are the detailed data requirements?

Developer-facing details (exact columns, download format, mapping rules) live in:
- `data_prep/README.md`
