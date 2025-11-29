"""
people_extend.py — Age–sex dependent initialization helpers for Starsim v3+
"""

import os
import numpy as np
import pandas as pd
import logging
import starsim as ss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def fixed_step_die(self):
    """
    Robust MIGHTI-compatible replacement for StarSim.People.step_die().

    - Ensures alive/dead flags are synchronized.
    - Preserves ti_dead/ti_removed semantics.
    - Notifies every disease module once per step.
    - Compatible with DeathsExtended, NonAcquiredDisease, and all base diseases.
    """
    ti = self.sim.ti
    ti_dead_raw = self.ti_dead.raw
    ti_removed_raw = self.ti_removed.raw

    # Determine who should die this step
    death_mask   = np.isfinite(ti_dead_raw)   & (ti_dead_raw   <= ti)
    removal_mask = np.isfinite(ti_removed_raw) & (ti_removed_raw <= ti)
    death_uids = np.where(death_mask | removal_mask)[0]

    if len(death_uids) == 0:
        return np.array([], dtype=int)
    
    # Ensure all UIDs are within valid population bounds
    n_people = len(self)
    valid_mask = death_uids < n_people
    death_uids = death_uids[valid_mask]
    
    if len(death_uids) == 0:
        return np.array([], dtype=int)

    # --- 1. Update state arrays (mirrors Starsim finalize_deaths) ---
    # Filter out already-dead individuals to avoid re-processing
    # Check both .raw and direct access to handle BoolState filtering
    try:
        already_dead_raw = self.dead.raw[death_uids] if hasattr(self.dead, 'raw') else None
        # Also check if they're already marked as dead in the main array
        # (BoolState might filter, so we need to check .raw)
        if already_dead_raw is not None:
            already_dead = already_dead_raw
        else:
            # Fallback: try direct access
            already_dead = np.array(self.dead[death_uids], dtype=bool) if hasattr(self.dead, '__getitem__') else np.zeros(len(death_uids), dtype=bool)
    except (IndexError, KeyError):
        # If indexing fails, assume none are dead yet
        already_dead = np.zeros(len(death_uids), dtype=bool)
    
    new_deaths = death_uids[~already_dead]
    
    if len(new_deaths) == 0:
        # All these deaths were already processed
        return np.array([], dtype=int)
    
    # Update state arrays for newly dead individuals
    # Use both .raw and direct assignment to ensure state is properly set
    self.alive.raw[new_deaths] = False
    self.dead.raw[new_deaths] = True
    # Also set via direct assignment if BoolState supports it
    try:
        self.dead[new_deaths] = True
    except (IndexError, KeyError, TypeError):
        # BoolState might not support direct indexing, that's okay
        pass
    
    self.ti_dead.raw[new_deaths] = np.minimum(
        np.where(np.isfinite(ti_dead_raw[new_deaths]), ti_dead_raw[new_deaths], ti),
        ti
    )
    # maintain parity between ti_dead and ti_removed
    self.ti_removed.raw[new_deaths] = self.ti_dead.raw[new_deaths]
    
    # Update death_uids to only include new deaths for notification
    death_uids = new_deaths

    # --- 2. Notify all diseases and mortality-linked modules ---
    for module in self.sim.module_list:
        if hasattr(module, "record_deaths"):
            module.record_deaths(death_uids)
        elif hasattr(module, "step_die"):
            module.step_die(death_uids)

    logger.debug(f"[MIGHTI step_die] committed {len(death_uids)} deaths at ti={ti}")
    return death_uids

ss.People.step_die = fixed_step_die

# ---------------------------------------------------------------------------
# 1. Build 5-year age–sex percent table
# ---------------------------------------------------------------------------
def build_age_sex_percent(csv_path: str, init_year: int,
                          out_csv: str | None = None,
                          *, bin_width: int = 5, top_open: int = 95) -> pd.DataFrame:
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
def build_age_distribution_from_percent(age_sex_csv: str, *, bin_width: int = 5, top_open: int = 95) -> pd.DataFrame:
    df = pd.read_csv(age_sex_csv)
    total_by_bin = df["male"] + df["female"]
    df["age"] = df["agestart"] + bin_width / 2
    df["value"] = total_by_bin / total_by_bin.sum()
    df["value"] /= df["value"].sum()
    return df[["age", "value"]]


# ---------------------------------------------------------------------------
# 3. Build p(female) mapping
# ---------------------------------------------------------------------------
def build_p_female_map(age_sex_df: pd.DataFrame, *, bin_width: int = 5, top_open: int = 95) -> dict:
    p_map = {}
    for _, row in age_sex_df.iterrows():
        a = int(row["agestart"])
        f, m = row["female"], row["male"]
        total = f + m
        label = f"{a}-{a+bin_width-1}" if a < top_open else f"{top_open}+"
        p_map[label] = f / total if total > 0 else 0.5
    return p_map


def set_p_by_age_factory(p_map: dict, *, bin_width: int = 5, top_open: int = 95):
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
def make_people_with_age_sex(csv_path: str, init_year: int, n_agents: int,
                             *, out_dir: str = "data_processed",
                             bin_width: int = 5, top_open: int = 95) -> ss.People:
    """
    Build Starsim-compatible People object using empirical age–sex distribution.
    Uses lower-bin ages (e.g., 0,5,10,...) for age_data to ensure correct sampling
    of youngest agents (0–4 bin not undercounted).
    """

    region_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_age_distribution", "")
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
    weights     = ((age_sex_df["male"] + age_sex_df["female"]) / 100.0).to_numpy()
    weights    /= weights.sum()

    # Instead of just storing the lower bin edge, sample uniformly within bin
    rng = np.random.default_rng(42)
    samples = []
    for lo, w in zip(ages_lower, weights):
        # draw roughly proportional number of samples (1e4 total as template)
        n = int(w * 1e4)
        samples.extend(rng.uniform(lo, lo + bin_width, n))
    samples = np.array(samples)

    # Build a DataFrame for Starsim age_data (lower edge only for histogram)
    age_df = pd.DataFrame({
        "age": ages_lower,
        "value": weights
    })
    age_df["value"] /= age_df["value"].sum()

    # Step 3: Build p(female)
    p_map = build_p_female_map(age_sex_df, bin_width=bin_width, top_open=top_open)

    # Step 4: Create People object (Starsim v3 automatically handles age sampling)
    ppl = ss.People(n_agents=n_agents, age_data=age_df)
    ppl.female.default.pars.p = set_p_by_age_factory(p_map, bin_width=bin_width, top_open=top_open)

    # Step 5: Debug sampling to confirm 0–4 proportion
    rng = np.random.default_rng(42)
    sampled_ages = np.random.choice(age_df["age"], size=10000, p=age_df["value"])
    hist, edges = np.histogram(sampled_ages, bins=list(range(0, 100, 5)))
    proportions = (hist / hist.sum()).round(4)
    sample_table = pd.DataFrame({
        "agebin": [f"{int(edges[i])}-{int(edges[i+1]-1)}" for i in range(len(hist))],
        "prop": proportions
    })

    return ppl


# # ---------------------------------------------------------------------------
# # 5. Extended Deaths class — ensures death requests are committed each step
# # ---------------------------------------------------------------------------
# class DeathsExtended(ss.Module):
#     """A demographic module that finalizes any requested deaths each step."""

#     def __init__(self, death_rate=None, rate_units=1, **kwargs):
#         super().__init__(**kwargs)
#         self.name = "deathsextended"
#         self.death_rate = death_rate
#         self.rate_units = rate_units

#     def step(self):
#         ppl = self.sim.people
#         death_uids = ppl.step_die()  # Finalize all requested deaths
#         if len(death_uids):
#             print(f"[DeathsExtended] committed {len(death_uids)} deaths at ti={self.sim.ti}")

#     def finalize(self):
#         super().finalize()

#     def finalize_results(self):
#         super().finalize_results()
