### `data_prep/` (developer tooling)

This folder contains scripts that transform **raw** datasets in `data/raw/` into
**MIGHTI-ready** inputs in `data/processed/`.

The modeling package itself should read only from `data/processed/` (or test
fixtures under `tests/test_data/`).

---

### Required disease file (GBD download) for `data_prep/data_cleaning.py`

`data_prep/data_cleaning.py` expects **one long-format prevalence CSV** per region, stored in:

- `data/raw/disease_data/{region}_disease_prevalence_agesex.csv`

Example for Eswatini:
- `data/raw/disease_data/eswatini_disease_prevalence_agesex.csv`

This file is used to generate:
- `data/processed/{region}_prevalence.csv`

---

### Exact required columns (must match these names)

The prevalence loader expects these columns to exist in the CSV:

- **`cause`**: condition name (GBD cause/risk string; must match entries in `data_prep/cleaning/paths.py:cause_map`)
- **`val`**: prevalence value (either **Percent** or **Fraction**; see below)
- **`sex`**: `Male` / `Female` (case-insensitive; `M`/`F` are also accepted)
- **`age`**: age-group label like `<5 years`, `5-9 years`, `10-14 years`, … (GBD-style strings)
- **`year`**: year as `YYYY` (or any string containing a 4-digit year; the loader extracts the first `\d{4}`)

Other columns can exist (e.g., `measure`, `location`, `metric`, `upper`, `lower`) and will be ignored.

---

### Value / unit expectations

In the GBD Results export you’ll often get prevalence in **Percent**.

The loader handles both cases:
- If `val` looks like **percent** (max value > 1.5), it will divide by 100 to convert to a fraction.
- Otherwise it assumes `val` is already a **fraction** in \([0,1]\).

---

### “Cause” naming requirements (what must match)

MIGHTI maps GBD labels to internal model condition names using:
- `data_prep/cleaning/paths.py` → `cause_map`

So your downloaded `cause` strings must match the keys in `cause_map` (examples: `Cardiovascular diseases`, `Road injuries`, `Alzheimer’s disease and other dementias`, `Influenza and pneumonia`, etc.).

If a `cause` is not in `cause_map`, it will be dropped during preprocessing (it will not appear in `data/processed/{region}_prevalence.csv`).

---

### Minimal GBD download guidance (what to select)

When downloading from IHME GBD, ensure your export includes:
- **Measure**: `Prevalence`
- **Dimensions**: `Location`, `Year`, `Age`, `Sex`, `Cause`
- **Metric/value**: a numeric prevalence value in the `val` column (Percent is OK)

Then save the exported CSV exactly as:
- `data/raw/disease_data/{region}_disease_prevalence_agesex.csv`

