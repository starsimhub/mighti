import numpy as np
import pandas as pd

REGION = "eswatini"
prev_path = f"data/processed/{REGION}_prevalence.csv"
age_path = f"data/processed/{REGION}_age_distribution.csv"
out_path = f"data/processed/{REGION}_postprocess_check_prevalence.csv"

prev = pd.read_csv(prev_path)
age = pd.read_csv(age_path)

# ---- prevalence table (Age bins x Year) ----
prev["Age"] = pd.to_numeric(prev["Age"], errors="coerce").astype(int)
prev["Year"] = pd.to_numeric(prev["Year"], errors="coerce").astype(int)

# collect diseases from *_male / *_female columns
male_cols = [c for c in prev.columns if c.endswith("_male")]
female_cols = [c for c in prev.columns if c.endswith("_female")]
diseases = sorted(set(c[:-5] for c in male_cols) & set(c[:-7] for c in female_cols))

# ---- age distribution table (single ages by sex, wide years) ----
age["age"] = pd.to_numeric(age["age"], errors="coerce").astype(int)
age["sex"] = age["sex"].astype(str).str.lower().str.strip()

year_cols = [c for c in age.columns if c.isdigit()]
age_long = age.melt(id_vars=["age", "sex"], value_vars=year_cols, var_name="Year", value_name="pop")
age_long["Year"] = age_long["Year"].astype(int)
age_long["pop"] = pd.to_numeric(age_long["pop"], errors="coerce")

# map single-age -> prevalence bin (0,5,...,80)
age_long["Age"] = (age_long["age"] // 5) * 5
age_long.loc[age_long["Age"] > 80, "Age"] = 80

# aggregate to Age-bin x Year x sex
w = (
    age_long.groupby(["Year", "sex", "Age"], as_index=False)["pop"]
    .sum()
    .dropna(subset=["pop"])
)

# If age distribution years are sparse (e.g. every 2 years), interpolate to all prevalence years
all_prev_years = sorted(prev["Year"].unique())
grid = pd.MultiIndex.from_product(
    [sorted(w["sex"].unique()), sorted(w["Age"].unique()), all_prev_years],
    names=["sex", "Age", "Year"]
).to_frame(index=False)

w_full = grid.merge(w, on=["sex", "Age", "Year"], how="left")
w_full["pop"] = (
    w_full.sort_values(["sex", "Age", "Year"])
    .groupby(["sex", "Age"], as_index=False)["pop"]
    .transform(lambda s: s.interpolate(limit_direction="both"))
)

# ---- weighted all-age prevalence by year/sex ----
rows = []
for sex in ["male", "female"]:
    for yr in all_prev_years:
        ww = w_full[(w_full["sex"] == sex) & (w_full["Year"] == yr)][["Age", "pop"]].copy()
        ww = ww.dropna(subset=["pop"])
        if ww.empty or ww["pop"].sum() <= 0:
            continue

        row = {"year": yr, "sex": sex.capitalize()}
        prev_y = prev[prev["Year"] == yr].set_index("Age")

        for d in diseases:
            col = f"{d}_{sex}"
            if col not in prev_y.columns:
                row[d] = np.nan
                continue

            tmp = ww.join(prev_y[[col]], on="Age", how="left")
            vals = pd.to_numeric(tmp[col], errors="coerce")
            pop = pd.to_numeric(tmp["pop"], errors="coerce")
            m = vals.notna() & pop.notna() & (pop > 0)

            if m.any():
                row[d] = float((vals[m] * pop[m]).sum() / pop[m].sum())
            else:
                row[d] = np.nan

        rows.append(row)

out = pd.DataFrame(rows).sort_values(["year", "sex"]).reset_index(drop=True)
out.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {out.shape}")
