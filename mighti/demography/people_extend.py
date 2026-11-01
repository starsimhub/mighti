"""
people_extend.py — Age–sex dependent initialization of the people
"""

import os
import numpy as np
import pandas as pd
import logging
import starsim as ss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# 1. Build 5-year age–sex percent table
# ---------------------------------------------------------------------------
def build_age_sex_percent(csv_path, init_year,
                          out_csv=None,
                          *, bin_width=5, top_open=95):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.map(str.strip)
    year_col = str(init_year)
    if year_col not in df.columns:
        raise ValueError(f"Year {init_year} not found in {csv_path}")

    df = df[["age", "sex", year_col]].rename(columns={year_col: "pop"})
    df["sex"] = df["sex"].astype(str)

    ages = np.sort(df["age"].unique())
    is_five_year = len(ages) > 1 and np.all(np.diff(ages) == 5) and ages[0] in (0, 5)

    bins = list(range(0, top_open + bin_width, bin_width)) + [np.inf]
    labels = [int(lo) for lo in range(0, top_open, bin_width)] + [top_open]

    if is_five_year:
        df["agestart"] = df["age"]
    else:
        df["agestart"] = pd.cut(
            df["age"], bins=bins, labels=labels, right=False, include_lowest=True
        ).astype(int)

    grouped = df.groupby(["agestart", "sex"], observed=True)["pop"].sum().reset_index()
    pivot = (grouped.pivot(index="agestart", columns="sex", values="pop")
                      .fillna(0)
                      .rename(columns=str.capitalize)
                      .rename(columns={"Male": "male", "Female": "female"})
                      .reset_index())

    total = pivot["male"].sum() + pivot["female"].sum()
    pivot["male"] = pivot["male"] / total * 100
    pivot["female"] = pivot["female"] / total * 100

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pivot.to_csv(out_csv, index=False)
    return pivot


# ---------------------------------------------------------------------------
# 2. Build Starsim-compatible age distribution
# ---------------------------------------------------------------------------
def build_age_distribution_from_percent(age_sex_csv, *, bin_width=5, top_open=95):
    df = pd.read_csv(age_sex_csv)
    total_by_bin = df["male"] + df["female"]
    df["age"] = df["agestart"] + bin_width / 2
    df["value"] = total_by_bin / total_by_bin.sum()
    df["value"] /= df["value"].sum()
    return df[["age", "value"]]


# ---------------------------------------------------------------------------
# 3. Build p(female) mapping
# ---------------------------------------------------------------------------
def build_p_female_map(age_sex_df, *, bin_width=5, top_open=95):
    p_map = {}
    for _, row in age_sex_df.iterrows():
        a = int(row["agestart"])
        f, m = row["female"], row["male"]
        total = f + m
        label = f"{a}-{a+bin_width-1}" if a < top_open else f"{top_open}+"
        p_map[label] = f / total if total > 0 else 0.5
    return p_map


def set_p_by_age_factory(p_map, *, bin_width=5, top_open=95):
    bins = list(range(0, top_open + bin_width, bin_width)) + [np.inf]
    labels = [f"{lo}-{lo+bin_width-1}" for lo in range(0, top_open, bin_width)] + [f"{top_open}+"]

    def set_p_by_age(sim, uids):
        ages = sim.people.age[uids]
        groups = pd.cut(ages, bins=bins, labels=labels, right=False)
        probs = np.array([p_map.get(str(g), 0.5) for g in groups])
        return probs

    return set_p_by_age


# ---------------------------------------------------------------------------
# 4. Unified builder for Starsim People 
# ---------------------------------------------------------------------------
def make_people_with_age_sex(csv_path, init_year, n_agents,
                             *, out_dir=None,
                             extra_states=None,
                             bin_width=5, top_open=95):
    """
    Build a Starsim `People` object from demographic inputs.

    Supported input formats
    -----------------------
    1) **Wide age–sex-by-year table** (raw input):
       Columns: `age`, `sex`, and a column named exactly `str(init_year)` with population counts.
       This format allows MIGHTI to derive an age distribution *and* an age-dependent sex assignment.

    2) **Starsim-ready age distribution** (already processed):
       Columns: `age`, `value` (where value is a weight/count/probability).
       In this case, we use the provided age distribution directly and leave the default
       sex assignment untouched (Starsim defaults apply).
    """

    # If the caller already provides an age distribution (age,value), use it directly.
    # This is what `prepare_data_for_year.prepare_data_for_year()` writes.
    peek = pd.read_csv(csv_path, nrows=5)
    cols = {c.strip() for c in peek.columns}
    if {"age", "value"}.issubset(cols) and "sex" not in cols:
        age_df = pd.read_csv(csv_path)
        age_df.columns = age_df.columns.map(str.strip)
        age_df = age_df[["age", "value"]].copy()
        age_df["age"] = pd.to_numeric(age_df["age"], errors="coerce")
        age_df["value"] = pd.to_numeric(age_df["value"], errors="coerce")
        age_df = age_df.dropna(subset=["age", "value"])
        v = age_df["value"].to_numpy(dtype=float)
        vsum = float(v.sum())
        if vsum > 0:
            age_df["value"] = v / vsum
        ppl = ss.People(n_agents=n_agents, age_data=age_df, extra_states=extra_states)
        return ppl

    region_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_age_distribution", "")
    out_csv = None
    if out_dir:
        out_csv = os.path.join(out_dir, f"{region_name}_age_sex_percent_{init_year}.csv")

    # Step 1: Export 5-year age–sex percent CSV
    age_sex_df = build_age_sex_percent(csv_path, init_year, out_csv,
                                       bin_width=bin_width, top_open=top_open)

    # # Step 2: Build age distribution (use lower edges for Starsim v3)
    # age_df = pd.DataFrame({
    #     "age": age_sex_df["agestart"],  # lower bin edge
    #     "value": (age_sex_df["male"] + age_sex_df["female"]) / 100.0
    # })
    # age_df["value"] /= age_df["value"].sum()

    # Step 2: Build age distribution (use uniform sampling within each bin)
    ages_lower = age_sex_df["agestart"].to_numpy()
    # NOTE: avoid in-place ops on arrays that may be read-only under pandas CoW/Arrow
    weights = ((age_sex_df["male"] + age_sex_df["female"]) / 100.0).to_numpy(copy=True).astype(float, copy=False)
    wsum = float(weights.sum())
    weights = weights / wsum if wsum > 0 else np.full_like(weights, 1.0 / len(weights))

    # Build a DataFrame for Starsim age_data (lower edge only for histogram)
    age_df = pd.DataFrame({
        "age": ages_lower,
        "value": weights
    })
    age_df["value"] = age_df["value"] / age_df["value"].sum()

    # Step 3: Build p(female)
    p_map = build_p_female_map(age_sex_df, bin_width=bin_width, top_open=top_open)

    # Step 4: Create People object (Starsim samples the ages)
    ppl = ss.People(n_agents=n_agents, age_data=age_df, extra_states=extra_states)
    ppl.female.default.pars.p = set_p_by_age_factory(p_map, bin_width=bin_width, top_open=top_open)

    return ppl
