### `data_prep/` (developer tooling)

This folder contains scripts that transform **raw** datasets in `data/raw/` into
**MIGHTI-ready** inputs in `data/processed/`.

The modeling package itself should read only from `data/processed/` (or test
fixtures under `tests/test_data/`).

---

### Inputs from IHME/GBD (what to download, what to save)

`data_prep/data_cleaning.py` expects **GBD long-format CSV exports** saved under:

- `data/raw/disease_data/`

It generates MIGHTI-ready inputs under:

- `data/processed/`

This README describes the recommended workflow going forward:

- **one multi-year prevalence file** per region (for calibration)
- **one baseline-year rate file** per region (for `p_death` and `dur_condition`)

---

## 1) Prevalence input (for calibration)

### What to download from GBD

- **Measure**: `Prevalence`
- **Metric**: `Percent`
- **Cause**: all causes you want to include in the model
- **Location**: specify location
- **Age**: `<5 years, 5-9 years, 10-14 years, 15-19 years, …, 80+ years`
- **Sex**: `Male`, `Female`
- **Year**: the full year range you want to use for calibration

### Where to save it

Save as:

- `data/raw/disease_data/{region}_disease_prevalence_agesex.csv`

Example:

- `data/raw/disease_data/eswatini_disease_prevalence_agesex.csv`

### Required columns (must match these names)

The prevalence loader uses:

- **`cause`**: GBD cause/risk string (must match keys in `data_prep/cleaning/paths.py:cause_map`)
- **`val`**: prevalence value (Percent; fraction is also accepted and auto-detected)
- **`sex`**: `Male` / `Female` (case-insensitive; `M`/`F` also accepted)
- **`age`**: GBD-style age-group labels (e.g., `<5 years`, `5-9 years`, …, `80+ years`)
- **`year`**: `YYYY` (or any string containing a 4-digit year)

Other columns may exist and are ignored (e.g., `measure`, `location`, `metric`, `upper`, `lower`, `population_group`).

### Units

Prevalence is stored internally as a **fraction in [0, 1]**.
If `val` looks like percent (max > 1.5), it is divided by 100.

---

## 2) Parameter input (for `p_death` and `dur_condition`)

### What to download from GBD

- **Measures**: `Deaths`, `Prevalence`, `Incidence`
- **Metric**: `Rate` (per 100,000)
- **Cause**: all causes you want to include in the model
- **Location**: specify location
- **Age**: `All ages`
- **Sex**: `Both` (preferred; Male/Female may also be included)
- **Year**: pick a single **baseline** year

### Baseline-year guidance

Choose a year that can reasonably be interpreted as **pre-intervention** for the intervention(s) you care about,
so that derived `p_death` and `dur_condition` approximate the “no-intervention” natural history.
Earlier is often better, but avoid years so early that they no longer reflect relevant dynamics.

### Where to save it

Save as:

- `data/raw/disease_data/allcause_rate.csv`

---

## 3) Derived parameter definitions (how preprocessing uses the baseline-year rate file)

- **Mean duration**:
  - `dur_condition = prevalence_per_100k / incidence_per_100k`

- **Mortality parameter** (depends on `disease_type` in the parameters table):
  - chronic/default: `p_death ≈ deaths_per_100k / prevalence_per_100k`
  - acute/event-like: `p_death ≈ deaths_per_100k / incidence_per_100k` (CFR-style proxy)

### Conditions modeled with no direct mortality (`p_death = 0`)

Some modeled conditions are treated as **non-fatal states** in MIGHTI (they do not directly cause deaths in the simulation; mortality effects are handled via other conditions or background mortality). For these conditions, preprocessing enforces:

- `p_death = 0`

Current list:

- `AnxietyDisorder`
- `BipolarDisorder`
- `ChronicPain`
- `Hyperlipidemia`
- `Hypertension`
- `Obesity`
- `TobaccoUse`

Important: leaving `p_death` blank/NA is not equivalent to zero in the current implementation; missing values fall back to a nonzero default during parameter loading.

---

## 4) Cause naming requirements

GBD cause strings are mapped to internal model condition names using:

- `data_prep/cleaning/paths.py` → `cause_map`

If a cause string is not in `cause_map`, it will be dropped during preprocessing.

