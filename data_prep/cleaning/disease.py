import os
import logging
import re

import numpy as np
import pandas as pd

from .paths import cause_map


logger = logging.getLogger(__name__)

# Conditions that are modeled without direct condition-caused mortality.
# IMPORTANT: Do NOT leave these p_death values blank/NA, because runtime parameter loading
# falls back to a nonzero default (see `mighti/diseases/base_disease.py:get_disease_parameters`).
NONMORTAL_P_DEATH_CONDITIONS = {
    "AnxietyDisorder",
    "BipolarDisorder",
    "ChronicPain",
    "Hyperlipidemia",
    "Hypertension",
    "Obesity",
    "TobaccoUse",
}


MIGHTI_PARAMETERS_COLUMNS = [
    "condition",
    "p_death",
    "dur_condition",
    "rel_sus",
    "remission_rate",
    "max_disease_duration",
    "p_acquire",
    "p_acquire_female",
    "p_acquire_male",
    "disease_type",
    "affected_sex",
]


MIGHTI_CONDITION_METADATA = [
    # Core NCDs + risks
    ("Type1Diabetes", "chronic", "both"),
    ("Type2Diabetes", "remitting", "both"),
    ("Hypertension", "chronic", "both"),
    ("Obesity", "chronic", "both"),
    ("CardiovascularDiseases", "chronic", "both"),
    ("ChronicKidneyDisease", "chronic", "both"),
    ("Hyperlipidemia", "chronic", "both"),
    ("CervicalCancer", "chronic", "female"),
    ("ColorectalCancer", "chronic", "both"),
    ("BreastCancer", "chronic", "both"),
    ("LungCancer", "chronic", "both"),
    ("ProstateCancer", "chronic", "male"),
    ("AlcoholUseDisorder", "remitting", "both"),
    ("TobaccoUse", "remitting", "both"),
    ("Dementia", "chronic", "both"),
    ("PTSD", "chronic", "both"),
    ("MajorDepressiveDisorder", "remitting", "both"),
    ("BipolarDisorder", "remitting", "both"),
    # Model hepatitis as an acute/event-like condition for Eswatini calibration inputs
    ("AcuteHepatitis", "acute", "both"),
    ("ChronicLiverDisease", "chronic", "both"),
    ("Asthma", "chronic", "both"),
    ("COPD", "chronic", "both"),
    ("AlzheimersDisease", "chronic", "both"),
    ("ParkinsonsDisease", "chronic", "both"),
    ("DrugUseDisorder", "remitting", "both"),
    ("AnxietyDisorder", "remitting", "both"),
    ("ChronicPain", "remitting", "both"),

    # Infectious / SIS-SIR placeholders used by MIGHTI
    ("HPV", "genericsis", "both"),
    ("Influenza", "genericsir", "both"),
    ("DiarrhealDiseases", "genericsir", "both"),

    # Injuries / violence
    ("InterpersonalViolence", "surgical", "both"),
    ("RoadInjuries", "surgical", "both"),
    ("SelfHarm", "acute", "both"),

    # Neonatal + congenital
    ("NeonatalEncephalopathy", "acute", "both"),
    ("NeonatalPretermBirth", "acute", "both"),
    ("NeonatalSepsis", "acute", "both"),
    ("NeonatalJaundice", "acute", "both"),
    ("CongenitalHeartAnomalies", "surgical", "both"),
    ("CongenitalMusculoskeletal", "surgical", "both"),
    ("DigestiveCongenitalAnomalies", "surgical", "both"),
    ("DownSyndrome", "static", "both"),
    ("NeuralTubeDefects", "surgical", "both"),
    ("ChromosomalAbnormalities", "static", "both"),
    ("EsophagealCancer", "chronic", "both"),
    ("ProteinEnergyMalnutrition", "acute", "both"),

    # New Eswatini mortality-driven additions
    ("COVID19", "genericsis", "both"),
    ("LowerRespiratoryInfections", "genericsis", "both"),
    ("Tuberculosis", "genericsir", "both"),
    ("MaternalConditions", "acute", "female"),
]


def apply_rel_sus_from_csv(
    parameters_csv_path,
    *,
    rel_sus_csv_path,
    output_csv_path=None,
    overwrite_existing=False,
):
    """
    Fill/overwrite `rel_sus` in parameters table using a 2-column CSV:
      - condition
      - rel_sus
    """
    out = output_csv_path or parameters_csv_path
    params = pd.read_csv(parameters_csv_path)
    rel = pd.read_csv(rel_sus_csv_path)

    if "condition" not in params.columns:
        raise ValueError(f"Missing 'condition' column in {parameters_csv_path}")
    if "condition" not in rel.columns or "rel_sus" not in rel.columns:
        raise ValueError(f"{rel_sus_csv_path} must contain columns: condition, rel_sus")

    params["condition"] = params["condition"].astype(str).str.strip()
    rel["condition"] = rel["condition"].astype(str).str.strip()
    rel["rel_sus"] = pd.to_numeric(rel["rel_sus"], errors="coerce")

    rel_map = rel.dropna(subset=["condition"]).drop_duplicates(subset=["condition"], keep="last").set_index("condition")["rel_sus"]
    params["rel_sus_calc"] = params["condition"].map(rel_map)

    if "rel_sus" not in params.columns:
        params["rel_sus"] = np.nan
    params["rel_sus"] = pd.to_numeric(params["rel_sus"], errors="coerce")

    if overwrite_existing:
        params["rel_sus"] = params["rel_sus_calc"].where(params["rel_sus_calc"].notna(), params["rel_sus"])
    else:
        missing = params["rel_sus"].isna()
        params.loc[missing, "rel_sus"] = params.loc[missing, "rel_sus_calc"]

    n_filled = int(params["rel_sus_calc"].notna().sum())
    params = params.drop(columns=["rel_sus_calc"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    params.to_csv(out, index=False)
    logger.info("Applied rel_sus from %s -> %s (%s mapped conditions)", rel_sus_csv_path, out, n_filled)
    return params


def apply_parameter_rules(
    parameters_csv_path,
    *,
    output_csv_path=None,
    overwrite_existing=False,
    max_duration_rules=None,
):
    """
    Apply rule-based defaults to parameter table:
      - remission_rate = 0 for non-remitting disease types
      - fill max_disease_duration by disease_type rule
    """
    out = output_csv_path or parameters_csv_path
    params = pd.read_csv(parameters_csv_path)

    if "condition" not in params.columns:
        raise ValueError(f"Missing 'condition' column in {parameters_csv_path}")
    if "disease_type" not in params.columns:
        raise ValueError(f"Missing 'disease_type' column in {parameters_csv_path}")

    if "remission_rate" not in params.columns:
        params["remission_rate"] = np.nan
    if "max_disease_duration" not in params.columns:
        params["max_disease_duration"] = np.nan

    params["disease_type_norm"] = params["disease_type"].astype(str).str.strip().str.lower()
    params["remission_rate"] = pd.to_numeric(params["remission_rate"], errors="coerce")
    params["max_disease_duration"] = pd.to_numeric(params["max_disease_duration"], errors="coerce")

    # User-facing rule: explicitly set non-remitting conditions to 0 remission
    non_remitting = params["disease_type_norm"] != "remitting"
    params.loc[non_remitting, "remission_rate"] = 0.0

    default_rules = {
        "acute": 1.0,
        "surgical": 1.0,
        "genericsis": 1.0,
        "genericsir": 1.0,
        "sis": 1.0,
        "sir": 1.0,
        "remitting": 10.0,
        "chronic": 100.0,
        "static": 120.0,
    }
    rules = default_rules if max_duration_rules is None else max_duration_rules

    for dtype, max_years in rules.items():
        m = params["disease_type_norm"] == str(dtype).lower()
        if overwrite_existing:
            params.loc[m, "max_disease_duration"] = float(max_years)
        else:
            missing = m & params["max_disease_duration"].isna()
            params.loc[missing, "max_disease_duration"] = float(max_years)

    params = params.drop(columns=["disease_type_norm"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    params.to_csv(out, index=False)
    logger.info("Applied remission/max-duration rules -> %s", out)
    return params


def create_mighti_parameters_template(output_csv, *, overwrite=False):
    """
    Create a template parameter CSV in the same schema as `mighti/data/eswatini_parameters.csv`.

    This is useful when you *do not* have a long-format file containing Deaths/Incidence
    to estimate `p_death` and `dur_condition`, but still want a complete parameter table
    for all modeled conditions (with blanks that can be filled later).
    """
    if (not overwrite) and os.path.exists(output_csv):
        logger.info(f"Parameter file already exists; keeping existing file: {output_csv}")
        return pd.read_csv(output_csv)

    df = pd.DataFrame(MIGHTI_CONDITION_METADATA, columns=["condition", "disease_type", "affected_sex"])

    # Fill required columns (leave numeric ones blank)
    for col in [
        "p_death",
        "dur_condition",
        "rel_sus",
        "remission_rate",
        "max_disease_duration",
        "p_acquire",
        "p_acquire_female",
        "p_acquire_male",
    ]:
        df[col] = ""

    # Force p_death=0 for conditions modeled without direct mortality
    df.loc[df["condition"].isin(NONMORTAL_P_DEATH_CONDITIONS), "p_death"] = 0.0

    df = df[MIGHTI_PARAMETERS_COLUMNS]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"MIGHTI parameter template saved to {output_csv}")
    return df


def ensure_conditions_in_parameters_csv(
    parameters_csv_path,
    *,
    output_csv_path=None,
    overwrite=True,
    create_if_missing=True,
    prune_extras=False,
):
    """
    Ensure `parameters_csv_path` contains at least the conditions in `MIGHTI_CONDITION_METADATA`.

    - Adds missing columns needed by MIGHTI
    - Appends missing conditions with blank numeric parameters
    - Preserves existing values for existing conditions
    - Optionally prunes conditions not in `MIGHTI_CONDITION_METADATA`
    """
    out = output_csv_path or parameters_csv_path

    if not os.path.exists(parameters_csv_path):
        if not create_if_missing:
            raise FileNotFoundError(parameters_csv_path)
        return create_mighti_parameters_template(out, overwrite=overwrite)

    df = pd.read_csv(parameters_csv_path)

    # Ensure required columns exist
    for col in MIGHTI_PARAMETERS_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Standardize condition key
    df["condition"] = df["condition"].astype(str).str.strip()

    # Append any missing conditions
    existing = set(df["condition"].dropna())
    rows = []
    for condition, disease_type, affected_sex in MIGHTI_CONDITION_METADATA:
        if condition in existing:
            continue
        row = {c: "" for c in MIGHTI_PARAMETERS_COLUMNS}
        row["condition"] = condition
        row["disease_type"] = disease_type
        row["affected_sex"] = affected_sex
        if condition in NONMORTAL_P_DEATH_CONDITIONS:
            row["p_death"] = 0.0
        rows.append(row)

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # Optionally drop conditions that are not modeled
    if prune_extras:
        modeled = {c for c, _, _ in MIGHTI_CONDITION_METADATA}
        df = df[df["condition"].isin(modeled)].copy()

    # Reorder and write
    df = df[MIGHTI_PARAMETERS_COLUMNS]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if overwrite or (not os.path.exists(out)):
        df.to_csv(out, index=False)
        logger.info(f"Updated parameter table saved to {out}")
    else:
        logger.info(f"Not overwriting existing file: {out}")
    return df


def fill_p_death_and_duration_from_allcause_files(
    parameters_csv_path,
    *,
    allcause_rate_csv_path,
    year=2007,
    output_csv_path=None,
    overwrite_existing=False,
):
    """
    Fill `p_death` and `dur_condition` in a MIGHTI parameters CSV using GBD-style
    all-cause exports.


    - `allcause_rate_csv_path`: Deaths/Prevalence/Incidence in "Rate" metric
      (per 100k).

    Computations (matching `data_cleaning_rev`):
      - p_death = deaths_per_100k / prevalence_rate_per_100k
      - dur_condition = prevalence_rate_per_100k / incidence_rate_per_100k

    Notes:
    - Only conditions present in the allcause files (after `cause_map`) can be filled.
    - If multiple raw causes map to the same modeled condition (many→one), we SUM them
      before computing.
    """
    # Keep a single implementation of the computation in `fill_p_death_and_duration_from_gbd_long_files()`
    # to avoid drift between multiple versions of the same logic.
    return fill_p_death_and_duration_from_gbd_long_files(
        parameters_csv_path=parameters_csv_path,
        gbd_long_csv_paths=[allcause_rate_csv_path],
        year=year,
        output_csv_path=output_csv_path,
        overwrite_existing=overwrite_existing,
    )


def fill_p_death_and_duration_from_gbd_long_files(
    parameters_csv_path,
    *,
    gbd_long_csv_paths,
    year=2007,
    location="Eswatini",
    output_csv_path=None,
    overwrite_existing=False,
    acute_duration_years_threshold=1.0,
):
    """
    Fill `p_death` and `dur_condition` using one or more GBD-style long-format exports
    that contain rows with columns like:

      measure, location, sex, age, cause, metric, year, val

    Some exports may include extra leading columns like `population_group`; these are ignored.

    Requirements for calculations:
    - Rates per 100k: metric='Rate' for measures Deaths/Prevalence/Incidence

    Computations:
      - dur_condition = prevalence_rate_per_100k / incidence_rate_per_100k

    p_death computation depends on disease type:
      - For acute/event-like conditions we prefer a CFR-style proxy:
          p_death = deaths_rate_per_100k / incidence_rate_per_100k
        This is typically much more stable than using point-prevalence, which can be tiny.
        We apply this when the condition's `disease_type` in the parameters table is one of:
          {"acute", "surgical", "genericsis", "genericsir", "sis", "sir"}
        (and when incidence is available and >0).

      - Otherwise (default, prevalence-based proxy):
          p_death = deaths_rate_per_100k / prevalence_rate_per_100k

    Note: `acute_duration_years_threshold` is kept for backward-compatibility/experimentation,
    but the primary switch is now `disease_type`.
    """
    out = output_csv_path or parameters_csv_path

    params = pd.read_csv(parameters_csv_path)
    if "condition" not in params.columns:
        raise ValueError(f"Missing 'condition' column in {parameters_csv_path}")

    for col in ["p_death", "dur_condition"]:
        if col not in params.columns:
            params[col] = np.nan

    params["condition"] = params["condition"].astype(str).str.strip()

    frames = []
    for p in gbd_long_csv_paths:
        df = pd.read_csv(p)
        # Allow extra columns; keep only what we need if present
        needed = ["measure", "location", "sex", "age", "cause", "metric", "year", "val"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{p} is missing required columns: {missing}")
        frames.append(df[needed].copy())
    raw = pd.concat(frames, ignore_index=True)

    # Basic filters
    raw = raw[
        (raw["location"] == location)
        & (raw["sex"] == "Both")
        & (raw["age"] == "All ages")
        & (raw["year"] == year)
    ].copy()

    raw["condition"] = raw["cause"].map(cause_map)
    raw = raw.dropna(subset=["condition"])

    # Determine which modeled conditions should use CFR for p_death (Deaths/Incidence).
    # We infer this from the parameter table's disease_type column.
    cfr_types = {"acute", "surgical", "genericsis", "genericsir", "sis", "sir"}
    cfr_conditions = set()
    if "disease_type" in params.columns:
        cfr_conditions = set(
            params.loc[params["disease_type"].astype(str).str.lower().isin(cfr_types), "condition"]
            .astype(str)
            .str.strip()
        )

    # Rates (Rate metric) — sum many→one mappings
    rates = raw[raw["metric"] == "Rate"].copy()
    deaths_rate = rates[rates["measure"] == "Deaths"].groupby("condition", as_index=True)["val"].sum()
    prev_rate = rates[rates["measure"] == "Prevalence"].groupby("condition", as_index=True)["val"].sum()
    inc_rate = rates[rates["measure"] == "Incidence"].groupby("condition", as_index=True)["val"].sum()

    idx = sorted(set(deaths_rate.index) | set(prev_rate.index) | set(inc_rate.index))
    computed = pd.DataFrame(index=idx)
    computed.index.name = "condition"
    computed["deaths_per_100k"] = deaths_rate.reindex(idx)
    computed["prevalence_rate_per_100k"] = prev_rate.reindex(idx)
    computed["incidence_rate_per_100k"] = inc_rate.reindex(idx)

    computed["dur_condition"] = np.nan
    ok2 = (
        computed["prevalence_rate_per_100k"].notna()
        & computed["incidence_rate_per_100k"].notna()
        & (computed["incidence_rate_per_100k"] > 0)
    )
    computed.loc[ok2, "dur_condition"] = computed.loc[ok2, "prevalence_rate_per_100k"] / computed.loc[ok2, "incidence_rate_per_100k"]

    # Default p_death (prevalence-based proxy)
    computed["p_death"] = np.nan
    ok_prev = (
        computed["deaths_per_100k"].notna()
        & computed["prevalence_rate_per_100k"].notna()
        & (computed["prevalence_rate_per_100k"] > 0)
    )
    computed.loc[ok_prev, "p_death"] = (
        computed.loc[ok_prev, "deaths_per_100k"] / computed.loc[ok_prev, "prevalence_rate_per_100k"]
    )

    # For CFR-eligible conditions, overwrite p_death with CFR proxy (deaths/incidence)
    cfr_applied_conditions = set()
    if cfr_conditions:
        is_cfr = computed.index.to_series().isin(cfr_conditions)
        ok_cfr = (
            is_cfr
            & computed["deaths_per_100k"].notna()
            & computed["incidence_rate_per_100k"].notna()
            & (computed["incidence_rate_per_100k"] > 0)
        )
        computed.loc[ok_cfr, "p_death"] = (
            computed.loc[ok_cfr, "deaths_per_100k"] / computed.loc[ok_cfr, "incidence_rate_per_100k"]
        )
        cfr_applied_conditions = set(computed.index[ok_cfr])

    # Keep p_death in [0, 1] when interpreted as a probability
    computed["p_death"] = computed["p_death"].clip(lower=0, upper=1)

    comp_short = computed[["p_death", "dur_condition"]].reset_index()
    merged = params.merge(comp_short, on="condition", how="left", suffixes=("", "_calc"))

    for col in ["p_death", "dur_condition"]:
        calc = f"{col}_calc"
        if col == "p_death" and cfr_applied_conditions:
            # Always overwrite p_death for CFR-eligible conditions if we computed it
            force = merged["condition"].isin(cfr_applied_conditions) & merged[calc].notna()
            merged.loc[force, col] = merged.loc[force, calc]
        if overwrite_existing:
            merged[col] = merged[calc].where(merged[calc].notna(), merged[col])
        else:
            missing = merged[col].isna()
            merged.loc[missing, col] = merged.loc[missing, calc]
        merged = merged.drop(columns=[calc])

    # Enforce p_death=0 for non-mortality conditions regardless of computed values
    if NONMORTAL_P_DEATH_CONDITIONS:
        merged.loc[merged["condition"].isin(NONMORTAL_P_DEATH_CONDITIONS), "p_death"] = 0.0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    merged.to_csv(out, index=False)
    logger.info(f"Filled p_death/dur_condition saved to {out} (year={year})")
    return merged


def create_and_fill_prevalence_template_from_long_format(
    raw_csv,
    output_csv,
    start_year=1987,
    end_year=2021,
    age_starts=None,
    *,
    overwrite=False,
):
    """
    Create and fill a prevalence template with numeric Age values (0, 5, 10, 15...).
    """

    if age_starts is None:
        # lower bounds for age groups, consistent with GBD
        age_starts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    # Step 1 — load raw prevalence file
    df = pd.read_csv(raw_csv).rename(columns={"cause": "condition", "val": "prevalence"})
    df["condition"] = df["condition"].map(cause_map)
    df = df.dropna(subset=["condition"])

    # normalize sex labels
    df["sex"] = df["sex"].str.lower().map({"male": "male", "m": "male", "female": "female", "f": "female"})

    # extract numeric year
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(float).astype("Int64")

    # normalize prevalence (% → fraction)
    if df["prevalence"].max() > 1.5:
        df["prevalence"] /= 100.0

    # If year range is not specified, infer from input file
    years = df["year"].dropna()
    if years.empty:
        raise ValueError(f"Could not parse any 4-digit years from: {raw_csv}")
    if start_year is None:
        start_year = int(years.min())
    if end_year is None:
        end_year = int(years.max())
    if end_year < start_year:
        raise ValueError(f"Invalid year range: start_year={start_year}, end_year={end_year}")

    # Step 2 — make the base grid (Age × Year, with one column per condition×sex)
    modeled_conditions = list(set(cause_map.values()))
    expected_cols = [f"{cond}_{sex}" for cond in modeled_conditions for sex in ["male", "female"]]
    grid = pd.MultiIndex.from_product([age_starts, range(start_year, end_year + 1)], names=["Age", "Year"]).to_frame(
        index=False
    )
    for c in expected_cols:
        grid[c] = np.nan

    # Step 3 — map GBD-style age labels to numeric start
    def get_age_start(label):
        if isinstance(label, str):
            label = label.strip()
            if re.match(r"^<\s*5", label):
                return 0
            m = re.match(r"^(\d+)", label)
            if m:
                return int(m.group(1))
        return np.nan

    df["Age"] = df["age"].apply(get_age_start)
    df = df.dropna(subset=["Age"])

    # Step 3.5 — aggregate collisions from cause_map many→one mappings
    # e.g. Hepatitis B/C → ViralHepatitis, multiple maternal sub-causes → MaternalConditions
    df = (
        df.groupby(["condition", "sex", "Age", "year"], as_index=False)["prevalence"]
        .sum()
    )

    # Step 4 — fill grid
    for (cond, sex), group in df.groupby(["condition", "sex"]):
        col = f"{cond}_{sex}"
        if col in grid.columns:
            for _, r in group.iterrows():
                mask = (grid["Age"] == r["Age"]) & (grid["Year"] == r["year"])
                grid.loc[mask, col] = r["prevalence"]

    # Step 5 — save (only if missing unless overwrite=True)
    if (not overwrite) and os.path.exists(output_csv):
        logger.info(f"Prevalence file already exists; keeping existing file: {output_csv}")
        return grid

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    grid.to_csv(output_csv, index=False)
    logger.info(f"Filled prevalence template saved to {output_csv}")
    return grid


def gbd_age_label_to_start(label):
    """Map a GBD age label to the numeric lower bound used in MIGHTI grids."""
    if isinstance(label, str):
        label = label.strip()
        if re.match(r"^<\s*5", label):
            return 0
        m = re.match(r"^(\d+)", label)
        if m:
            return int(m.group(1))
    return np.nan


def create_death_rate_grid_from_gbd_long(
    raw_csv,
    output_csv,
    *,
    location="Eswatini",
    start_year=None,
    end_year=None,
    age_starts=None,
    overwrite=False,
):
    """
    Build a wide death-rate grid (per 100k) with columns Age, Year, {Condition}_male/female.

    Input must be GBD long format with measure=Deaths and metric=Rate.
    """
    if age_starts is None:
        age_starts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    df = pd.read_csv(raw_csv)
    needed = ["measure", "location", "sex", "age", "cause", "metric", "year", "val"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{raw_csv} is missing required columns: {missing}")

    df = df[
        (df["measure"] == "Deaths")
        & (df["metric"] == "Rate")
        & (df["location"] == location)
    ].copy()
    if df.empty:
        raise ValueError(f"No Deaths/Rate rows for location={location!r} in {raw_csv}")

    df = df.rename(columns={"cause": "condition", "val": "death_rate"})
    df["condition"] = df["condition"].map(cause_map)
    df = df.dropna(subset=["condition"])
    df["sex"] = df["sex"].str.lower().map({"male": "male", "m": "male", "female": "female", "f": "female"})
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(float).astype("Int64")
    df["Age"] = df["age"].apply(gbd_age_label_to_start)
    df = df.dropna(subset=["Age"])
    df["Age"] = df["Age"].astype(int)

    years = df["year"].dropna()
    if years.empty:
        raise ValueError(f"Could not parse any 4-digit years from: {raw_csv}")
    if start_year is None:
        start_year = int(years.min())
    if end_year is None:
        end_year = int(years.max())

    modeled_conditions = sorted(df["condition"].unique())
    expected_cols = [f"{cond}_{sex}" for cond in modeled_conditions for sex in ["male", "female"]]
    grid = pd.MultiIndex.from_product([age_starts, range(start_year, end_year + 1)], names=["Age", "Year"]).to_frame(
        index=False
    )
    for c in expected_cols:
        grid[c] = np.nan

    df = df.groupby(["condition", "sex", "Age", "year"], as_index=False)["death_rate"].sum()

    for (cond, sex), group in df.groupby(["condition", "sex"]):
        col = f"{cond}_{sex}"
        if col not in grid.columns:
            continue
        for _, r in group.iterrows():
            mask = (grid["Age"] == r["Age"]) & (grid["Year"] == r["year"])
            grid.loc[mask, col] = r["death_rate"]

    if (not overwrite) and os.path.exists(output_csv):
        logger.info(f"Death-rate file already exists; keeping existing file: {output_csv}")
        return grid

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    grid.to_csv(output_csv, index=False)
    logger.info(f"Death-rate grid saved to {output_csv}")
    return grid


def _population_weights_for_gbd_ages(age_dist_csv, year, location=None):
    """
    Build population weights for (age_start, sex) cells using WPP single-age counts.

    For a GBD age label with lower bound `a0`, weight = sum of population ages [a0, a0+5)
    (or to 100 for open-ended 80+).
    """
    df = pd.read_csv(age_dist_csv)
    ycol = str(year)
    if ycol not in df.columns:
        raise ValueError(f"Year {year} not found in age distribution file: {age_dist_csv}")

    weights = []
    for _, row in df.iterrows():
        age = int(row["age"])
        sex = str(row["sex"]).strip()
        pop = float(row[ycol])
        if pop <= 0:
            continue
        weights.append({"age": age, "sex": sex, "pop": pop})

    pop_df = pd.DataFrame(weights)
    if pop_df.empty:
        raise ValueError(f"No population weights parsed from {age_dist_csv}")

    def weight_for_cell(age_start, sex_label):
        sex_key = "Male" if str(sex_label).lower().startswith("m") else "Female"
        if age_start >= 80:
            sub = pop_df[(pop_df["sex"] == sex_key) & (pop_df["age"] >= 80)]
        else:
            sub = pop_df[(pop_df["sex"] == sex_key) & (pop_df["age"] >= age_start) & (pop_df["age"] < age_start + 5)]
        return float(sub["pop"].sum()) if not sub.empty else 0.0

    return weight_for_cell


def aggregate_gbd_deaths_to_both_all_ages(
    death_long_csv,
    *,
    year,
    location="Eswatini",
    age_dist_csv=None,
):
    """
    Aggregate age/sex-stratified GBD death rates to Both / All ages for parameter fill.

  Population-weighted mean across age-sex cells. Prefer a direct IHME Both/All ages export
    when available; this is an approximation when only fine-stratified exports exist.
    """
    df = pd.read_csv(death_long_csv)
    needed = ["measure", "location", "sex", "age", "cause", "metric", "year", "val"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{death_long_csv} is missing required columns: {missing}")

    df = df[
        (df["measure"] == "Deaths")
        & (df["metric"] == "Rate")
        & (df["location"] == location)
        & (pd.to_numeric(df["year"], errors="coerce") == year)
    ].copy()
    if df.empty:
        raise ValueError(f"No Deaths rows for {location} year={year} in {death_long_csv}")

    if age_dist_csv and os.path.exists(age_dist_csv):
        weight_fn = _population_weights_for_gbd_ages(age_dist_csv, year)
        df["w"] = df.apply(
            lambda r: weight_fn(gbd_age_label_to_start(r["age"]), r["sex"]),
            axis=1,
        )
    else:
        logger.warning("No age distribution for weights; using uniform weights for death aggregation")
        df["w"] = 1.0

    df = df[df["w"] > 0].copy()
    rows = []
    for cause, g in df.groupby("cause"):
        val = np.average(g["val"].astype(float), weights=g["w"].astype(float))
        rows.append(
            {
                "measure": "Deaths",
                "location": location,
                "sex": "Both",
                "age": "All ages",
                "cause": cause,
                "metric": "Rate",
                "year": year,
                "val": val,
                "upper": np.nan,
                "lower": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    logger.info(
        "Aggregated %d causes to Both/All ages for year=%s (from %s)",
        len(out),
        year,
        death_long_csv,
    )
    return out


def merge_deaths_into_allcause_rate(allcause_csv_path, aggregated_deaths_df, output_csv_path=None):
    """
    Replace Deaths rows in allcause_rate.csv for causes present in aggregated_deaths_df.
    """
    out = output_csv_path or allcause_csv_path
    base = pd.read_csv(allcause_csv_path)
    causes_to_update = set(aggregated_deaths_df["cause"].astype(str))
    keep = ~((base["measure"] == "Deaths") & (base["cause"].isin(causes_to_update)))
    merged = pd.concat([base.loc[keep], aggregated_deaths_df], ignore_index=True)
    merged.to_csv(out, index=False)
    logger.info(
        "Merged %d updated Deaths rows into %s (causes: %d)",
        len(aggregated_deaths_df),
        out,
        len(causes_to_update),
    )
    return merged


def merge_gbd_long_csvs(input_csv_paths, output_csv_path, *, required_cols=None):
    """
    Merge one or more GBD-style long-format CSV exports (same schema) into a single CSV.

    This is useful when a location's GBD export was downloaded in multiple batches (e.g., by year ranges).
    We do not attempt to re-compute values here; we simply concatenate and de-duplicate exact rows.
    """
    if required_cols is None:
        required_cols = ["measure", "location", "sex", "age", "cause", "metric", "year", "val", "upper", "lower"]

    frames = []
    for p in input_csv_paths:
        df = pd.read_csv(p)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{p} is missing required columns: {missing}")
        frames.append(df[required_cols].copy())

    merged = pd.concat(frames, ignore_index=True).drop_duplicates()
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    merged.to_csv(output_csv_path, index=False)
    logger.info(f"Merged {len(input_csv_paths)} GBD exports → {output_csv_path} (rows={len(merged)})")
    return output_csv_path


def create_condition_metadata_table(long_csv_path, output_csv):
    # Load raw data
    raw = pd.read_csv(long_csv_path)

    # Filter for relevant rows
    raw = raw[(raw["sex"] == "Both") & (raw["age"] == "All ages") & (raw["year"] == 1999)]

    # Pivot the data (use mean in case of duplicates)
    pivot = (
        raw.pivot_table(index="cause", columns="measure", values="val", aggfunc="mean")
        .rename_axis(None, axis=1)
        .reset_index()
    )

    # Confirm required columns exist
    required = ["Deaths", "Prevalence", "Incidence"]
    for col in required:
        if col not in pivot.columns:
            raise ValueError(f"Missing required column: '{col}' in pivoted data.")

    # Compute derived parameters
    # When inputs are all in Rate metric (per 100k), convert population rates to a conditional probability:
    #   p_death ≈ (deaths per 100k) / (prevalence per 100k)
    pivot["p_death"] = pivot["Deaths"] / pivot["Prevalence"]
    pivot["dur_condition"] = pivot["Prevalence"] / pivot["Incidence"]

    pivot["condition"] = pivot["cause"].map(cause_map)
    pivot = pivot.dropna(subset=["condition"])

    # Define fixed metadata for all conditions (schema aligned with mighti/data/*_parameters.csv)
    df_meta = pd.DataFrame(MIGHTI_CONDITION_METADATA, columns=["condition", "disease_type", "affected_sex"])

    # Merge estimates into metadata
    merged = df_meta.merge(pivot[["condition", "p_death", "dur_condition"]], on="condition", how="left")

    # Add empty columns for other parameters
    merged["rel_sus"] = ""
    merged["remission_rate"] = ""
    merged["max_disease_duration"] = ""
    merged["p_acquire"] = ""
    merged["p_acquire_female"] = ""
    merged["p_acquire_male"] = ""

    # Reorder columns
    merged = merged[MIGHTI_PARAMETERS_COLUMNS]

    # Round numeric columns
    merged["p_death"] = pd.to_numeric(merged["p_death"], errors="coerce").round(5)
    merged["dur_condition"] = pd.to_numeric(merged["dur_condition"], errors="coerce").round(2)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info(f"Condition metadata table saved to {output_csv}")

    return merged


def extract_prevalence_timeseries_by_sex(long_csv_path, output_csv):
    # Load raw file
    df = pd.read_csv(long_csv_path)

    # Filter
    df = df[(df["measure"] == "Prevalence") & (df["age"] == "All ages")]

    df["condition"] = df["cause"].map(cause_map)
    df = df.dropna(subset=["condition"])

    # Pivot: rows = (year, sex), columns = conditions
    wide = df.pivot_table(index=["year", "sex"], columns="condition", values="val", aggfunc="mean").reset_index()

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    wide.to_csv(output_csv, index=False)
    logger.info(f"Prevalence time series saved to {output_csv}")
    return wide

