"""
Build MIGHTI-ready region data for US/NYC settings.

Outputs (into `mighti/data/` by default):
  - {region}_mx.csv                 (wide: Age, Sex, <year columns>)
  - {region}_age_distribution.csv   (wide: age, sex, <year columns>)
  - {region}_prevalence.csv         (tidy grid: Age, Year, <condition_sex columns>)
  - {region}_asfr.csv               (fertility_rate table; minimal placeholder if missing)

Design goals:
  - Never overwrite existing output files unless overwrite=True
  - If raw sources are missing, generate a *template* with 0.0 values so scripts run
  - Allow HIV prevalence to come from non-GBD sources via a drop-in CSV
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildPaths:
    project_root: Path
    out_dir: Path
    raw_dir: Path

    def out(self, name: str) -> Path:
        return self.out_dir / name

    def raw(self, name: str) -> Path:
        return self.raw_dir / name


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths(region: str) -> BuildPaths:
    root = _project_root()
    return BuildPaths(
        project_root=root,
        out_dir=root / "mighti" / "data",
        raw_dir=root / "raw_data" / "us_data" / region,
    )


def _write_if_missing(path: Path, df: pd.DataFrame, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        logger.info("Keeping existing file: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote: %s", path)


def _wide_from_long_mx(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Expect long mx with columns: Age, Sex, Time, mx
    Output wide: Age, Sex, <year columns>
    """
    df = df_long.copy()
    df.columns = [c.strip() for c in df.columns]
    required = {"Age", "Sex", "Time", "mx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"mx long file missing columns: {sorted(missing)}")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").astype("Int64")
    df["mx"] = pd.to_numeric(df["mx"], errors="coerce")
    df = df.dropna(subset=["Time", "mx", "Age", "Sex"])
    wide = df.pivot_table(index=["Age", "Sex"], columns="Time", values="mx", aggfunc="mean").reset_index()
    # make column names strings to match existing expectations
    wide.columns = [str(c) if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    return wide


def _wide_from_long_age_dist(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Expect long population with columns: age, sex, year, value (or pop)
    Output wide: age, sex, <year columns>
    """
    df = df_long.copy()
    df.columns = [c.strip() for c in df.columns]
    # Accept either `value` or `pop`
    val_col = "value" if "value" in df.columns else ("pop" if "pop" in df.columns else None)
    if val_col is None:
        raise ValueError("age distribution long file must have 'value' or 'pop' column")
    required = {"age", "sex", "year", val_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"age distribution long file missing columns: {sorted(missing)}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["year", val_col, "age", "sex"])
    wide = df.pivot_table(index=["age", "sex"], columns="year", values=val_col, aggfunc="sum").reset_index()
    wide.columns = [str(c) if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    return wide


def _parse_census_age_group(label: str) -> tuple[int, int] | None:
    """Parse labels like 'Under 5 years', '5 to 9 years', '85 years and over' into inclusive (lo, hi)."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    s = str(label).strip().lstrip(".").strip()
    s = s.replace("years", "").replace("year", "").strip()
    if s.lower().startswith("total"):
        return None
    if "under" in s.lower():
        # Under 5
        nums = [int(x) for x in s.split() if x.isdigit()]
        if nums:
            return (0, max(nums[0] - 1, 0))
        return (0, 4)
    if "and over" in s.lower() or "+" in s:
        # 85 years and over
        nums = [int("".join([ch for ch in s if ch.isdigit()]) or 85)]
        return (nums[0], 100)
    if "to" in s:
        parts = [p.strip() for p in s.split("to")]
        try:
            lo = int("".join([ch for ch in parts[0] if ch.isdigit()]))
            hi = int("".join([ch for ch in parts[1] if ch.isdigit()]))
            return (lo, hi)
        except Exception:
            return None
    # single age?
    digits = "".join([ch for ch in s if ch.isdigit()])
    if digits:
        a = int(digits)
        return (a, a)
    return None


def _age_distribution_long_from_sc_est_excel(excel_path: Path, *, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Convert a Census SC-EST AGESEX Excel (state-level) into long age distribution.

    Output columns: age, sex, year, value (population counts)
    Expands 5-year age groups uniformly into single ages.
    """
    df = pd.read_excel(excel_path, sheet_name=0, header=None)
    # Header rows are consistent for these files:
    # row 3: years in certain columns; row 4: sex labels repeated
    year_row = df.iloc[3]
    sex_row = df.iloc[4]

    # Build list of (col_idx, year, sex)
    col_meta = []
    current_year = None
    for j in range(len(sex_row)):
        y = year_row.iloc[j]
        if pd.notna(y):
            current_year = int(y)
        sex = sex_row.iloc[j]
        if current_year is None or pd.isna(sex):
            continue
        sex_s = str(sex).strip()
        if sex_s not in ("Male", "Female"):
            continue
        if start_year <= current_year <= end_year:
            col_meta.append((j, current_year, sex_s))

    if not col_meta:
        raise ValueError(f"Could not identify year/sex columns in {excel_path}")

    # Data start at row 5, col 0 has age group labels
    out_rows = []
    for i in range(5, len(df)):
        label = df.iloc[i, 0]
        rng = _parse_census_age_group(label)
        if rng is None:
            continue
        lo, hi = rng
        width = hi - lo + 1
        for (j, year, sex) in col_meta:
            val = df.iloc[i, j]
            if pd.isna(val):
                continue
            pop = float(val)
            # spread uniformly across single ages in the group
            per_age = pop / width if width > 0 else pop
            for age in range(lo, hi + 1):
                out_rows.append({"age": age, "sex": sex, "year": year, "value": per_age})

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError(f"No population rows parsed from {excel_path}")
    # aggregate in case of overlapping labels
    out = out.groupby(["age", "sex", "year"], as_index=False)["value"].sum()
    return out


def _make_placeholder_mx(*, start_year: int, end_year: int) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    ages = list(range(0, 101))
    rows = []
    for sex in ["Male", "Female"]:
        base = 0.001 if sex == "Female" else 0.0012
        for age in ages:
            # crude Gompertz-ish placeholder: grows with age
            mx = min(0.5, base * np.exp(age / 18))
            row = {"Age": age, "Sex": sex}
            row.update({str(y): float(mx) for y in years})
            rows.append(row)
    return pd.DataFrame(rows)


def _make_placeholder_age_distribution(*, start_year: int, end_year: int) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    ages = list(range(0, 101))
    # simple stationary-ish distribution
    weights = np.exp(-np.array(ages) / 35.0)
    weights = weights / weights.sum()
    rows = []
    for sex in ["Male", "Female"]:
        for age, w in zip(ages, weights):
            row = {"age": age, "sex": sex}
            row.update({str(y): float(w) for y in years})
            rows.append(row)
    return pd.DataFrame(rows)


def _make_prevalence_template(*, start_year: int, end_year: int, conditions: list[str]) -> pd.DataFrame:
    """
    Output tidy prevalence grid:
      Age, Year, <cond>_male, <cond>_female (all 0.0 initially)
    """
    age_starts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    grid = pd.MultiIndex.from_product([age_starts, range(start_year, end_year + 1)], names=["Age", "Year"]).to_frame(index=False)
    for cond in conditions:
        grid[f"{cond}_male"] = 0.0
        grid[f"{cond}_female"] = 0.0
    return grid


def _merge_hiv_prevalence(prevalence_df: pd.DataFrame, hiv_long: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a local HIV prevalence by age/sex/year and merge into the template.
    Expected columns in hiv_long:
      - Age, Year, HIV_male, HIV_female   (fractions 0..1 or %)
    """
    df = hiv_long.copy()
    df.columns = [c.strip() for c in df.columns]
    required = {"Age", "Year", "HIV_male", "HIV_female"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HIV prevalence raw missing columns: {sorted(missing)}")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["HIV_male"] = pd.to_numeric(df["HIV_male"], errors="coerce")
    df["HIV_female"] = pd.to_numeric(df["HIV_female"], errors="coerce")
    df = df.dropna(subset=["Age", "Year", "HIV_male", "HIV_female"])
    # normalize percent to fraction
    if max(df["HIV_male"].max(), df["HIV_female"].max()) > 1.5:
        df[["HIV_male", "HIV_female"]] = df[["HIV_male", "HIV_female"]] / 100.0
    out = prevalence_df.copy()
    out = out.merge(df[["Age", "Year", "HIV_male", "HIV_female"]], on=["Age", "Year"], how="left", suffixes=("", "_new"))
    for c in ["HIV_male", "HIV_female"]:
        newc = f"{c}_new"
        if newc in out.columns:
            out[c] = out[newc].combine_first(out[c])
            out = out.drop(columns=[newc])
    return out


def ensure_region_data(
    *,
    region: str,
    start_year: int,
    end_year: int,
    overwrite: bool = False,
    paths: BuildPaths | None = None,
) -> dict:
    """
    Ensure core data inputs exist for a region. Never overwrites unless overwrite=True.

    Raw input conventions (optional):
      raw_data/us_data/{region}/mx_long.csv
      raw_data/us_data/{region}/age_distribution_long.csv
      raw_data/us_data/{region}/hiv_prevalence_by_age_sex.csv
    """
    paths = paths or _default_paths(region)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) mx.csv
    mx_out = paths.out(f"{region}_mx.csv")
    mx_long = paths.raw("mx_long.csv")
    if not mx_out.exists() or overwrite:
        if mx_long.exists():
            wide = _wide_from_long_mx(pd.read_csv(mx_long))
        else:
            logger.warning("Missing raw mx_long.csv for %s; generating placeholder mx", region)
            wide = _make_placeholder_mx(start_year=start_year, end_year=end_year)
        _write_if_missing(mx_out, wide, overwrite=overwrite)
    else:
        logger.info("mx exists: %s", mx_out)

    # 2) age_distribution.csv
    age_out = paths.out(f"{region}_age_distribution.csv")
    age_long = paths.raw("age_distribution_long.csv")
    if not age_out.exists() or overwrite:
        if age_long.exists():
            wide = _wide_from_long_age_dist(pd.read_csv(age_long))
        else:
            # Try Census SC-EST state agesex excel if present
            excel_candidate = None
            for cand in [
                paths.raw("sc-est2024-agesex-36.xlsx"),
                paths.project_root / "raw_data" / "us_data" / "sc-est2024-agesex-36.xlsx",
            ]:
                if cand.exists():
                    excel_candidate = cand
                    break

            if excel_candidate is not None:
                logger.warning("Using Census state agesex Excel for %s age distribution: %s", region, excel_candidate)
                long_df = _age_distribution_long_from_sc_est_excel(excel_candidate, start_year=start_year, end_year=end_year)
                wide = _wide_from_long_age_dist(long_df)
            else:
                logger.warning("Missing raw age_distribution_long.csv for %s; generating placeholder age distribution", region)
                wide = _make_placeholder_age_distribution(start_year=start_year, end_year=end_year)
        _write_if_missing(age_out, wide, overwrite=overwrite)
    else:
        logger.info("age distribution exists: %s", age_out)

    # 3) prevalence.csv (template + optional HIV merge)
    prev_out = paths.out(f"{region}_prevalence.csv")
    # keep the template aligned with what the codebase has modules for (+ HIV)
    default_conditions = [
        "HIV",
        "Type1Diabetes",
        "Type2Diabetes",
        "Hypertension",
        "Hyperlipidemia",
        "Obesity",
        "CardiovascularDiseases",
        "ChronicKidneyDisease",
        "ChronicLiverDisease",
        "COPD",
        "Asthma",
        "Dementia",
        "AlzheimersDisease",
        "PTSD",
        "MajorDepressiveDisorder",
        "AlcoholUseDisorder",
        "TobaccoUse",
        "RoadInjuries",
        "InterpersonalViolence",
        "Flu",
        "HPV",
        "ViralHepatitis",
    ]
    if not prev_out.exists() or overwrite:
        prev = _make_prevalence_template(start_year=start_year, end_year=end_year, conditions=default_conditions)
        hiv_long = paths.raw("hiv_prevalence_by_age_sex.csv")
        if hiv_long.exists():
            prev = _merge_hiv_prevalence(prev, pd.read_csv(hiv_long))
        else:
            logger.info("No HIV prevalence drop-in found for %s (raw_data/us_data/%s/hiv_prevalence_by_age_sex.csv)", region, region)
        _write_if_missing(prev_out, prev, overwrite=overwrite)
    else:
        logger.info("prevalence exists: %s", prev_out)

    # 4) ASFR placeholder (so scripts don't crash; replace with real NYC fertility later)
    asfr_out = paths.out(f"{region}_asfr.csv")
    if not asfr_out.exists() or overwrite:
        # Minimal structure: columns likely read by ss.Pregnancy(pars={'fertility_rate': df})
        # Keep it non-empty but effectively zero fertility.
        asfr = pd.DataFrame({"Time": [start_year], "AgeGrp": [15], "ASFR": [0.0]})
        _write_if_missing(asfr_out, asfr, overwrite=overwrite)
    else:
        logger.info("asfr exists: %s", asfr_out)

    # Manifest for provenance
    manifest = {
        "region": region,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "out_dir": str(paths.out_dir),
        "raw_dir": str(paths.raw_dir),
        "outputs": {
            "mx": str(mx_out),
            "age_distribution": str(age_out),
            "prevalence": str(prev_out),
            "asfr": str(asfr_out),
        },
        "raw_inputs_present": {
            "mx_long": str(mx_long) if mx_long.exists() else None,
            "age_distribution_long": str(age_long) if age_long.exists() else None,
            "hiv_prevalence_by_age_sex": str(paths.raw("hiv_prevalence_by_age_sex.csv")) if paths.raw("hiv_prevalence_by_age_sex.csv").exists() else None,
        },
    }
    manifest_path = paths.project_root / "data_prep" / "us_data" / "manifests" / f"{region}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest: %s", manifest_path)
    return manifest

