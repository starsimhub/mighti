"""
Calibrate disease acquisition (p_acquire_multiplier) for specified conditions using
MIGHTI prevalence data. Uses disease classes from diseases_for_calibration (p_acquire=1
so the calibrated parameter is p_acquire_multiplier). Outputs best-fit parameter and
writes results to calibration_results_<Condition>.txt and calibrated_p_acquire.csv.
"""

import argparse
import os
from pathlib import Path
import optuna
import matplotlib.pyplot as plt
import numpy as np
import mighti as mi
import pandas as pd
import sciris as sc
import starsim as ss
import stisim as sti
from importlib import import_module
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Config
region = "eswatini"
init_year = 2007
end_year = 2023
total_trials = 500  # Joint (p_acquire, p_death) fit; override via --total-trials or --smoke
WEIGHT_PREVALENCE = float(os.environ.get("MIGHTI_CALIB_WEIGHT_PREV", "1.0"))
WEIGHT_DEATH = float(os.environ.get("MIGHTI_CALIB_WEIGHT_DEATH", "1.0"))
# +/- 1 order of magnitude around GBD-derived p_death seed (set via --pdeath-bound-mult)
P_DEATH_BOUND_MULT = float(os.environ.get("MIGHTI_CALIB_PDEATH_BOUND_MULT", "10.0"))
# Whether to include p_death in the Optuna search; toggled via --fit-pdeath / --no-fit-pdeath.
FIT_PDEATH = os.environ.get("MIGHTI_CALIB_FIT_PDEATH", "1") not in {"0", "false", "False", ""}
# Conditions that are modeled with no direct mortality; keep p_death=0 and skip the new dim.
NONMORTAL_P_DEATH_CONDITIONS = {
    "AnxietyDisorder",
    "BipolarDisorder",
    "ChronicPain",
    "Hyperlipidemia",
    "Hypertension",
    "Obesity",
    "TobaccoUse",
}
HIV_BETA_M2F = 0.01688952663716571
HIV_BETA_M2C = 0.0444149203530297

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../MIGHTI/
DATA_DIR = REPO_ROOT / "data" / "processed"


def _resolve_data_file(filename):
    """
    Resolve a region input CSV within the repo.

    Preferred location is `data/processed/`. For developer/test workflows, we also
    allow falling back to `tests/test_data/`.
    """
    candidates = [
        REPO_ROOT / "data" / "processed" / filename,
        REPO_ROOT / "tests" / "test_data" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename!r}. Tried:\n- " + "\n- ".join(str(c) for c in candidates)
    )

# Paths
path_prevalence = str(_resolve_data_file(f"{region}_prevalence.csv"))
path_prevalence_hiv = str(_resolve_data_file(f"{region}_prevalence_hiv.csv"))
path_parameters = str(_resolve_data_file(f"{region}_parameters.csv"))
path_fertility = str(_resolve_data_file(f"{region}_asfr.csv"))
path_mortality = str(_resolve_data_file(f"{region}_mortality_rates.csv"))

def _try_resolve_death_rates():
    try:
        return pd.read_csv(_resolve_data_file(f"{region}_death_rates.csv"))
    except FileNotFoundError:
        logger.warning("Death-rate targets not found; calibration will use prevalence only")
        return None


death_df = _try_resolve_death_rates()

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# Results under package dir so path works from repo root or from mighti/calibration/
results_dir = REPO_ROOT / "mighti" / "calibration" / "results" / f"calibration_ver2_{region}_{date_str}"

# Load prevalence once for eval
prev_df = pd.read_csv(path_prevalence)
param_df = pd.read_csv(path_parameters)
param_df["condition"] = param_df["condition"].astype(str).str.strip()
if "affected_sex" in param_df.columns:
    param_df["affected_sex"] = param_df["affected_sex"].astype(str).str.strip().str.lower()
else:
    param_df["affected_sex"] = "both"

# Default conditions to calibrate when --conditions is not supplied:
# all conditions listed in region parameters. Downstream guards will skip
# conditions without usable prevalence data or missing disease classes.
all_conditions = param_df["condition"].dropna().unique().tolist()
conditions = all_conditions


OPTUNA_DIAGNOSTIC_METHODS = [
    "plot_optimization_history",
    "plot_param_importances",
    "plot_timeline",
    "plot_edf",
]

OPTUNA_DIAGNOSTIC_METHODS_PDEATH = OPTUNA_DIAGNOSTIC_METHODS + ["plot_contour"]

OPTUNA_PANEL_METHODS = [
    "plot_optimization_history",
    "fit",
    "plot_timeline",
    "plot_edf",
]

PANEL_TITLES = {
    "plot_optimization_history": "Optimization History",
    "fit": "Observed Vs Simulated Prevalence",
    "plot_timeline": "Trial Timeline",
    "plot_edf": "Empirical Distribution",
}


def _as_figure(obj):
    """
    Return a matplotlib Figure for objects returned by Optuna plotting.

    Depending on library versions, plot functions may return:
    - Figure
    - Axes
    - ndarray of Axes (e.g. plot_contour with 3+ params returns a grid)
    """
    if obj is None:
        return None
    if hasattr(obj, "savefig"):  # Figure-like
        return obj
    if hasattr(obj, "get_figure"):  # Axes-like
        try:
            return obj.get_figure()
        except Exception:
            return None
    # ndarray / list / nested-array of Axes
    try:
        arr = np.asarray(obj)
        flat = arr.flatten()
        for item in flat:
            if item is None:
                continue
            if hasattr(item, "get_figure"):
                try:
                    return item.get_figure()
                except Exception:
                    continue
            if hasattr(item, "savefig"):
                return item
    except Exception:
        pass
    return None


def save_optuna_diagnostics(calib, disease_name, out_dir, methods=None):
    """
    Save a compact, per-disease set of Optuna diagnostic figures.

    This uses Starsim's Calibration.plot_optuna() wrapper and writes one PNG
    per available method under:
      results/.../diagnostics/<DiseaseName>/
    """
    methods = methods or OPTUNA_DIAGNOSTIC_METHODS
    diag_dir = out_dir / "diagnostics" / disease_name
    diag_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for method in methods:
        try:
            # Run one method at a time so filename always matches the method.
            out = calib.plot_optuna(methods=[method])
            plot_obj = out[0] if out else None
        except Exception as e:
            logger.warning("Could not generate diagnostic %s for %s: %s", method, disease_name, e)
            continue

        fig = _as_figure(plot_obj)
        if fig is None:
            logger.warning("Skipping diagnostic %s for %s: unsupported plot object type", method, disease_name)
            continue

        out_png = diag_dir / f"{method}.png"
        try:
            fig.savefig(out_png, dpi=180, bbox_inches="tight")
            saved += 1
        except Exception as e:
            logger.warning("Failed to save diagnostic plot %s for %s: %s", method, disease_name, e)
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass

    if saved == 0:
        logger.warning("No Optuna diagnostics saved for %s", disease_name)
    else:
        logger.info("Saved %d Optuna diagnostics for %s -> %s", saved, disease_name, diag_dir)

def save_diagnostic_panel_from_pngs(disease_name, out_dir, panel_methods=None):
    """
    Build a 2x2 combined diagnostic panel from already-saved PNG files.

    This lets us compose `diagnostic_<Condition>.png` without rerunning calibration.
    The `fit` panel uses the observed-vs-simulated prevalence overlay.
    """
    panel_methods = panel_methods or OPTUNA_PANEL_METHODS
    diag_dir = out_dir / "diagnostics" / disease_name
    diag_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = np.ravel(axes)

    has_any = False
    for ax, method in zip(axes, panel_methods):
        if method == "fit":
            png = diag_dir / f"fit_{disease_name}.png"
        else:
            png = diag_dir / f"{method}.png"
        if png.exists():
            try:
                img = plt.imread(png)
                ax.imshow(img)
                has_any = True
            except Exception as e:
                logger.warning("Could not read %s for panel (%s): %s", method, disease_name, e)
                ax.text(0.5, 0.5, "Failed to load", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
        ax.set_title(PANEL_TITLES.get(method, method.replace("plot_", "").replace("_", " ").title()))
        ax.axis("off")

    for ax in axes[len(panel_methods):]:
        ax.axis("off")

    if not has_any:
        plt.close(fig)
        logger.warning("No diagnostic PNGs available to compose panel for %s", disease_name)
        return

    fig.suptitle(f"Calibration diagnostics: {disease_name}", fontsize=14)
    fig.tight_layout()
    out_png = diag_dir / f"diagnostic_{disease_name}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnostic panel for %s -> %s", disease_name, out_png)


def _get_prevalence_analyzer(sim):
    for analyzer in sim.analyzers.values():
        if isinstance(analyzer, (mi.analyzers.PrevalenceAnalyzer, mi.analyzers.PrevalenceAnalyzer_HIV)):
            return analyzer
    raise KeyError("No PrevalenceAnalyzer found in sim.analyzers")


def _get_death_analyzer(sim):
    for analyzer in sim.analyzers.values():
        if isinstance(analyzer, mi.analyzers.CauseDeathRateAnalyzer):
            return analyzer
    return None


def _normalized_mae(sim_vals, obs_vals):
    """Mean absolute error scaled by mean observed magnitude (avoid zero divide)."""
    sim_vals = np.asarray(sim_vals, dtype=float)
    obs_vals = np.asarray(obs_vals, dtype=float)
    mask = np.isfinite(sim_vals) & np.isfinite(obs_vals)
    if not np.any(mask):
        return 0.0
    scale = float(np.nanmean(np.abs(obs_vals[mask])))
    if scale <= 0:
        scale = 1.0
    return float(np.nanmean(np.abs(sim_vals[mask] - obs_vals[mask])) / scale)


def _result_to_array(analyzer, key, n_t):
    val = analyzer.results.get(key, None)
    if val is None:
        return np.zeros(n_t, dtype=float)
    if hasattr(val, "values"):
        val = val.values
    arr = np.asarray(val, dtype=float).ravel()
    if arr.size == n_t:
        return arr
    if arr.size == 1:
        return np.full(n_t, float(arr[0]), dtype=float)
    if arr.size < n_t:
        return np.pad(arr, (0, n_t - arr.size), mode="edge")
    return arr[:n_t]


def save_calibration_fit_plot(calib, disease_name, out_dir, observed_df, affected_sex="both"):
    """
    Save observed-vs-simulated prevalence fit plot per calibrated condition.
    """
    if getattr(calib, "after_msim", None) is None or not getattr(calib.after_msim, "sims", None):
        logger.warning("No calibrated simulation available to plot fit for %s", disease_name)
        return

    sim = calib.after_msim.sims[0]
    analyzer = _get_prevalence_analyzer(sim)
    n_t = len(sim.timevec)

    age_bins = getattr(analyzer, "age_bins", [])
    n_bins = len(age_bins)
    if n_bins == 0:
        logger.warning("No age bins found in prevalence analyzer for %s", disease_name)
        return

    years = pd.to_datetime(analyzer.timevec).year.to_numpy(dtype=int)
    dkey = disease_name.lower()

    male_num = np.zeros(n_t, dtype=float)
    male_den = np.zeros(n_t, dtype=float)
    female_num = np.zeros(n_t, dtype=float)
    female_den = np.zeros(n_t, dtype=float)
    male_dens_by_bin = []
    female_dens_by_bin = []

    for i in range(n_bins):
        male_num_i = _result_to_array(analyzer, f"{dkey}_num_male_{i}", n_t)
        male_den_i = _result_to_array(analyzer, f"{dkey}_den_male_{i}", n_t)
        female_num_i = _result_to_array(analyzer, f"{dkey}_num_female_{i}", n_t)
        female_den_i = _result_to_array(analyzer, f"{dkey}_den_female_{i}", n_t)
        male_num += male_num_i
        male_den += male_den_i
        female_num += female_num_i
        female_den += female_den_i
        male_dens_by_bin.append(male_den_i)
        female_dens_by_bin.append(female_den_i)

    sim_prev_m = np.divide(male_num, np.where(male_den > 0, male_den, 1.0))
    sim_prev_f = np.divide(female_num, np.where(female_den > 0, female_den, 1.0))

    # Build weighted all-age observed prevalence using simulation age-bin denominators as weights.
    obs = observed_df.copy()
    obs["Year"] = pd.to_numeric(obs["Year"], errors="coerce").astype("Int64")
    obs["Age"] = pd.to_numeric(obs["Age"], errors="coerce")
    mcol = f"{disease_name}_male"
    fcol = f"{disease_name}_female"
    if mcol not in obs.columns or fcol not in obs.columns:
        logger.warning("Observed prevalence columns missing for %s (%s, %s)", disease_name, mcol, fcol)
        return

    by_year = {
        int(y): g.set_index("Age")
        for y, g in obs.dropna(subset=["Year"]).groupby("Year")
    }

    obs_prev_m = np.full(n_t, np.nan, dtype=float)
    obs_prev_f = np.full(n_t, np.nan, dtype=float)
    age_lows = [a0 for a0, _ in age_bins]

    for ti, y in enumerate(years):
        g = by_year.get(int(y), None)
        if g is None:
            continue
        m_vals = []
        m_wts = []
        f_vals = []
        f_wts = []
        for bi, a0 in enumerate(age_lows):
            if a0 not in g.index:
                continue
            mv = pd.to_numeric(g.loc[a0, mcol], errors="coerce")
            fv = pd.to_numeric(g.loc[a0, fcol], errors="coerce")
            mw = float(male_dens_by_bin[bi][ti])
            fw = float(female_dens_by_bin[bi][ti])
            if pd.notna(mv) and mw > 0:
                m_vals.append(float(mv))
                m_wts.append(mw)
            if pd.notna(fv) and fw > 0:
                f_vals.append(float(fv))
                f_wts.append(fw)
        if m_wts:
            obs_prev_m[ti] = np.average(m_vals, weights=m_wts)
        if f_wts:
            obs_prev_f[ti] = np.average(f_vals, weights=f_wts)

    diag_dir = out_dir / "diagnostics" / disease_name
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_png = diag_dir / f"fit_{disease_name}.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    if affected_sex in {"male", "both"}:
        ax.plot(years, sim_prev_m * 100.0, color="blue", lw=2, label="Male simulated")
        ax.scatter(years, obs_prev_m * 100.0, color="blue", s=20, alpha=0.85, label="Male observed")
    if affected_sex in {"female", "both"}:
        ax.plot(years, sim_prev_f * 100.0, color="red", lw=2, label="Female simulated")
        ax.scatter(years, obs_prev_f * 100.0, color="red", s=20, alpha=0.85, label="Female observed")
    ax.set_title(f"Calibration fit: {disease_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Prevalence (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration fit plot for %s -> %s", disease_name, out_png)


def save_calibration_death_fit_plot(calib, disease_name, out_dir, observed_death_df, affected_sex="both"):
    """Save observed-vs-simulated cause-specific death rate (per 100k) plot."""
    if observed_death_df is None:
        return
    if getattr(calib, "after_msim", None) is None or not getattr(calib.after_msim, "sims", None):
        return

    sim = calib.after_msim.sims[0]
    death_an = _get_death_analyzer(sim)
    if death_an is None:
        return

    prev_an = _get_prevalence_analyzer(sim)
    n_t = len(sim.timevec)
    years = pd.to_datetime(prev_an.timevec).year.to_numpy(dtype=int)
    age_bins = getattr(prev_an, "age_bins", [])
    dkey = disease_name.lower()
    mcol = f"{disease_name}_male"
    fcol = f"{disease_name}_female"
    if mcol not in observed_death_df.columns or fcol not in observed_death_df.columns:
        return

    obs = observed_death_df.copy()
    obs["Year"] = pd.to_numeric(obs["Year"], errors="coerce").astype("Int64")
    obs["Age"] = pd.to_numeric(obs["Age"], errors="coerce")
    by_year = {int(y): g.set_index("Age") for y, g in obs.dropna(subset=["Year"]).groupby("Year")}

    sim_m = []
    sim_f = []
    obs_m = []
    obs_f = []
    plot_years = []

    for ti, y in enumerate(years):
        g = by_year.get(int(y))
        if g is None:
            continue
        plot_years.append(int(y))
        m_rates = []
        f_rates = []
        om = []
        of_ = []
        for i, (a0, _) in enumerate(age_bins):
            deaths_m = _result_to_array(death_an, f"{dkey}_deaths_male_{i}", n_t)
            deaths_f = _result_to_array(death_an, f"{dkey}_deaths_female_{i}", n_t)
            expo_m = _result_to_array(death_an, f"{dkey}_expo_male_{i}", n_t)
            expo_f = _result_to_array(death_an, f"{dkey}_expo_female_{i}", n_t)
            rate_m = deaths_m[ti] / max(expo_m[ti], 1.0) * 1e5
            rate_f = deaths_f[ti] / max(expo_f[ti], 1.0) * 1e5
            m_rates.append(rate_m)
            f_rates.append(rate_f)
            if a0 in g.index:
                om.append(pd.to_numeric(g.loc[a0, mcol], errors="coerce"))
                of_.append(pd.to_numeric(g.loc[a0, fcol], errors="coerce"))
        sim_m.append(np.nanmean(m_rates) if m_rates else np.nan)
        sim_f.append(np.nanmean(f_rates) if f_rates else np.nan)
        obs_m.append(np.nanmean(om) if om else np.nan)
        obs_f.append(np.nanmean(of_) if of_ else np.nan)

    if not plot_years:
        return

    diag_dir = out_dir / "diagnostics" / disease_name
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_png = diag_dir / f"fit_death_{disease_name}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    yrs = np.asarray(plot_years)
    if affected_sex in {"male", "both"}:
        ax.plot(yrs, sim_m, color="blue", lw=2, label="Male simulated")
        ax.scatter(yrs, obs_m, color="blue", s=20, alpha=0.85, label="Male observed")
    if affected_sex in {"female", "both"}:
        ax.plot(yrs, sim_f, color="red", lw=2, label="Female simulated")
        ax.scatter(yrs, obs_f, color="red", s=20, alpha=0.85, label="Female observed")
    ax.set_title(f"Death-rate fit: {disease_name} (per 100k)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Death rate (per 100,000)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved death-rate fit plot for %s -> %s", disease_name, out_png)


def has_observed_death_data(disease_name, data, affected_sex):
    """Return True when observed cause-specific death rates are usable for calibration."""
    if data is None:
        return False
    female_col = f"{disease_name}_female"
    male_col = f"{disease_name}_male"
    required_cols = []
    if affected_sex in {"female", "both"}:
        required_cols.append(female_col)
    if affected_sex in {"male", "both"}:
        required_cols.append(male_col)
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        return False
    obs = data[required_cols].apply(pd.to_numeric, errors="coerce")
    if not obs.notna().any().any():
        return False
    return bool((obs.fillna(0.0).to_numpy() > 0).any())


def has_observed_prevalence_data(disease_name, data):
    """
    Fast guard to skip calibration when observed prevalence is not usable.
    """
    female_col = f"{disease_name}_female"
    male_col = f"{disease_name}_male"

    missing_cols = [c for c in [female_col, male_col] if c not in data.columns]
    if missing_cols:
        logger.warning("Skipping %s: missing prevalence column(s): %s", disease_name, ", ".join(missing_cols))
        return False

    obs = data[[female_col, male_col]].apply(pd.to_numeric, errors="coerce")
    if not obs.notna().any().any():
        logger.warning("Skipping %s: prevalence columns are all missing/NaN", disease_name)
        return False

    if not (obs.fillna(0.0).to_numpy() > 0).any():
        logger.warning("Skipping %s: prevalence is all zero (no calibration signal)", disease_name)
        return False

    return True


def get_affected_sex(disease_name):
    """Return affected_sex metadata for a condition from parameters table."""
    row = param_df.loc[param_df["condition"] == str(disease_name).strip()]
    if row.empty:
        return "both"
    val = str(row.iloc[0].get("affected_sex", "both")).strip().lower()
    return val if val in {"male", "female", "both"} else "both"


def has_observed_prevalence_data_for_affected_sex(disease_name, data, affected_sex):
    """
    Fast guard for observed prevalence availability, respecting affected_sex.
    """
    female_col = f"{disease_name}_female"
    male_col = f"{disease_name}_male"

    required_cols = []
    if affected_sex in {"female", "both"}:
        required_cols.append(female_col)
    if affected_sex in {"male", "both"}:
        required_cols.append(male_col)

    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        logger.warning(
            "Skipping %s: missing prevalence column(s) for affected_sex=%s: %s",
            disease_name,
            affected_sex,
            ", ".join(missing_cols),
        )
        return False

    obs = data[required_cols].apply(pd.to_numeric, errors="coerce")
    if not obs.notna().any().any():
        logger.warning("Skipping %s: prevalence columns are all missing/NaN", disease_name)
        return False

    if not (obs.fillna(0.0).to_numpy() > 0).any():
        logger.warning("Skipping %s: prevalence is all zero (no calibration signal)", disease_name)
        return False

    return True


def try_get_condition_class(name):
    """Get disease class from diseases_for_calibration (p_acquire=1, calibrate p_acquire_multiplier)."""
    try:
        module = import_module("mighti.calibration.diseases_for_calibration")
        return getattr(module, name)
    except AttributeError:
        logger.warning("Skipping %s: not in diseases_for_calibration", name)
        return None


def make_sim(disease_name, DiseaseClass):
    """
    Build a sim for calibrating one disease: HIV + single condition from
    diseases_for_calibration. dt=1 (annual), p_acquire=1 in disease so we fit p_acquire_multiplier.
    """
    diseases = ["HIV", disease_name]
    disease_prevalence_df = pd.read_csv(path_prevalence)
    hiv_prevalence_df = pd.read_csv(path_prevalence_hiv)
    disease_prevalence_data, disease_age_bins = mi.initialize_prevalence_data(
        diseases=[disease_name],
        prevalence_data=disease_prevalence_df,
        inityear=init_year,
    )
    hiv_prevalence_data, hiv_age_bins = mi.initialize_prevalence_data(
        diseases=["HIV"],
        prevalence_data=hiv_prevalence_df,
        inityear=init_year,
    )
    prevalence_data = {**hiv_prevalence_data, **disease_prevalence_data}
    age_bins = {**hiv_age_bins, **disease_age_bins}

    def get_prevalence_function(d):
        def prevalence_func(sim, uids, size=None):
            return mi.age_sex_dependent_prevalence(
                disease=d,
                prevalence_data=prevalence_data,
                age_bins=age_bins,
                sim=sim,
                uids=uids,
                size=size,
            )
        return prevalence_func

    # HIV - use calibrated STIsim transmission parameters and MIGHTI age/sex-specific initial prevalence.
    hiv = sti.HIV(
        beta_m2f=HIV_BETA_M2F,
        beta_m2c=HIV_BETA_M2C,
        init_prev=ss.bernoulli(
            p=lambda sim, uids, size=None: get_prevalence_function("HIV")(sim, uids, size)
        ),
    )
    hiv.pars.include_aids_deaths = True
    hiv.pars.p_hiv_death = ss.bernoulli(p=0.00015)
    hiv.pars.include_care = True
    hiv.pars.art_efficacy = 0.9

    # Disease under calibration — from diseases_for_calibration (p_acquire=1)
    init_prev = ss.bernoulli(p=get_prevalence_function(disease_name))
    health_condition = DiseaseClass(csv_path=path_parameters, pars={"init_prev": init_prev})

    # Demographics
    death_rates = {"death_rate": pd.read_csv(path_mortality), "rate_units": 1}
    death = ss.Deaths(pars=death_rates)
    fertility_rate = {"fertility_rate": pd.read_csv(path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    networks = [ss.MaternalNet(), sti.StructuredSexual()]

    # Prevalence analyzer (HIV-stratified to match calibration_original)
    prevalence_analyzer = mi.analyzers.PrevalenceAnalyzer_HIV(
        prevalence_data=prevalence_data,
        diseases=diseases,
    )
    death_analyzer = mi.analyzers.CauseDeathRateAnalyzer(
        diseases=[disease_name],
        age_bins=prevalence_analyzer.age_bins,
    )

    sim = ss.Sim(
        dt=1,
        n_agents=10_000,
        total_pop=9_980_999,
        start=init_year,
        stop=end_year,
        diseases=[hiv, health_condition],
        networks=networks,
        demographics=[pregnancy, death],
        analyzers=[prevalence_analyzer, death_analyzer],
        copy_inputs=False,
        label=f"Calibration - {disease_name}",
    )
    sim.init()
    return sim


def run_calibration(disease_name, DiseaseClass):
    """
    Run calibration for one disease. Uses closures so build_sim and eval_fn
    see the correct disease_name (avoids wrong value when looping over conditions).
    """
    orig_disease_name = disease_name
    affected_sex = get_affected_sex(orig_disease_name)

    def build_sim_local(sim, calib_pars):
        if isinstance(sim, ss.MultiSim):
            sim = sim.sims[0]
        hc = sim.diseases[orig_disease_name.lower()]
        for k, pars in calib_pars.items():
            if k == "rand_seed":
                sim.pars.rand_seed = pars
                continue
            v = pars["value"]
            if k == "hc_p_death":
                # p_death is wrapped in ss.bernoulli at construction; preserve the
                # wrapper and just retune the underlying probability so .filter()
                # and .pars['p'] access patterns continue to work.
                try:
                    hc.pars.p_death.pars["p"] = float(v)
                except (AttributeError, KeyError, TypeError):
                    hc.pars.p_death = ss.bernoulli(float(v))
                continue
            if "hc_" in k:
                hc.pars[k.replace("hc_", "")] = v
        return sim

    use_death_objective = has_observed_death_data(orig_disease_name, death_df, affected_sex)

    def fit_by_sex_local(sim, prev_data, death_data=None):
        if isinstance(sim, ss.MultiSim):
            sim = sim.sims[0]
        female_prev_fit = 0.0
        male_prev_fit = 0.0
        female_death_fit = 0.0
        male_death_fit = 0.0
        prev_analyzer = None
        prev_label = None
        for label, analyzer in sim.analyzers.items():
            if isinstance(analyzer, (mi.analyzers.PrevalenceAnalyzer, mi.analyzers.PrevalenceAnalyzer_HIV)):
                prev_analyzer = analyzer
                prev_label = label
                break
        if prev_analyzer is None or prev_label is None:
            raise KeyError("No PrevalenceAnalyzer in sim.analyzers")
        prev_results = sim.results[prev_label]
        death_analyzer = _get_death_analyzer(sim)
        dkey = disease_name.lower()

        sim_prev_f = []
        sim_prev_m = []
        obs_prev_f = []
        obs_prev_m = []
        sim_death_f = []
        sim_death_m = []
        obs_death_f = []
        obs_death_m = []

        for index, (age_low, _) in enumerate(prev_analyzer.age_bins):
            obs = prev_data[prev_data["Age"] == age_low][
                ["Year", "Age", f"{orig_disease_name}_female", f"{orig_disease_name}_male"]
            ].copy()
            obs["Year"] = obs["Year"].astype(int)
            year_sim = pd.to_datetime(prev_analyzer.timevec).year
            sim_df = pd.DataFrame({
                "Year": year_sim.astype(int),
                "Age": age_low,
                "sim_female": prev_results[f"{dkey}_prev_female_{index}"],
                "sim_male": prev_results[f"{dkey}_prev_male_{index}"],
            })
            merged = pd.merge(obs, sim_df, on=["Year", "Age"], how="inner")
            if not merged.empty:
                if affected_sex in {"female", "both"}:
                    female_prev_fit += (merged["sim_female"] - merged[f"{orig_disease_name}_female"]).abs().sum()
                    sim_prev_f.extend(merged["sim_female"].tolist())
                    obs_prev_f.extend(merged[f"{orig_disease_name}_female"].tolist())
                if affected_sex in {"male", "both"}:
                    male_prev_fit += (merged["sim_male"] - merged[f"{orig_disease_name}_male"]).abs().sum()
                    sim_prev_m.extend(merged["sim_male"].tolist())
                    obs_prev_m.extend(merged[f"{orig_disease_name}_male"].tolist())

            if use_death_objective and death_data is not None and death_analyzer is not None:
                obs_d = death_data[death_data["Age"] == age_low][
                    ["Year", "Age", f"{orig_disease_name}_female", f"{orig_disease_name}_male"]
                ].copy()
                obs_d["Year"] = obs_d["Year"].astype(int)
                n_t = len(year_sim)
                sim_death_df = pd.DataFrame({
                    "Year": year_sim.astype(int),
                    "Age": age_low,
                    "sim_female": [
                        _result_to_array(death_analyzer, f"{dkey}_deaths_female_{index}", n_t)[ti]
                        / max(_result_to_array(death_analyzer, f"{dkey}_expo_female_{index}", n_t)[ti], 1.0)
                        * 1e5
                        for ti in range(n_t)
                    ],
                    "sim_male": [
                        _result_to_array(death_analyzer, f"{dkey}_deaths_male_{index}", n_t)[ti]
                        / max(_result_to_array(death_analyzer, f"{dkey}_expo_male_{index}", n_t)[ti], 1.0)
                        * 1e5
                        for ti in range(n_t)
                    ],
                })
                merged_d = pd.merge(obs_d, sim_death_df, on=["Year", "Age"], how="inner")
                if not merged_d.empty:
                    if affected_sex in {"female", "both"}:
                        female_death_fit += (merged_d["sim_female"] - merged_d[f"{orig_disease_name}_female"]).abs().sum()
                        sim_death_f.extend(merged_d["sim_female"].tolist())
                        obs_death_f.extend(merged_d[f"{orig_disease_name}_female"].tolist())
                    if affected_sex in {"male", "both"}:
                        male_death_fit += (merged_d["sim_male"] - merged_d[f"{orig_disease_name}_male"]).abs().sum()
                        sim_death_m.extend(merged_d["sim_male"].tolist())
                        obs_death_m.extend(merged_d[f"{orig_disease_name}_male"].tolist())

        prev_norm = _normalized_mae(
            sim_prev_f + sim_prev_m,
            obs_prev_f + obs_prev_m,
        )
        death_norm = 0.0
        if use_death_objective:
            death_norm = _normalized_mae(
                sim_death_f + sim_death_m,
                obs_death_f + obs_death_m,
            )
        return (
            float(female_prev_fit),
            float(male_prev_fit),
            float(female_death_fit),
            float(male_death_fit),
            prev_norm,
            death_norm,
        )

    def eval_fn_local(sim, data=None, **kwargs):
        (
            _ff,
            _fm,
            _df,
            _dm,
            prev_norm,
            death_norm,
        ) = fit_by_sex_local(sim, data, death_df if use_death_objective else None)
        return float(WEIGHT_PREVALENCE * prev_norm + WEIGHT_DEATH * death_norm)

    sim = make_sim(orig_disease_name, DiseaseClass)

    # Seed p_death from the parameters CSV; joint Optuna fit anchors on this seed.
    try:
        seed_pdeath_series = param_df.loc[
            param_df["condition"] == orig_disease_name, "p_death"
        ]
        seed_pdeath = float(seed_pdeath_series.iloc[0]) if len(seed_pdeath_series) else 0.0
    except Exception:
        seed_pdeath = 0.0

    is_nonmortal = orig_disease_name in NONMORTAL_P_DEATH_CONDITIONS
    fit_pdeath = (
        FIT_PDEATH
        and use_death_objective
        and (seed_pdeath > 0)
        and (not is_nonmortal)
    )

    def _p_acq_bounds():
        # Return a fresh dict each call: Starsim's Calibration mutates these in place
        # (sets "value"), so sharing a single dict across male/female would dedupe them.
        return dict(low=1e-6, high=0.1, guess=0.001, log=True)

    calib_pars = {}
    if affected_sex in {"female", "both"}:
        calib_pars["hc_p_acquire_multiplier_female"] = _p_acq_bounds()
    if affected_sex in {"male", "both"}:
        calib_pars["hc_p_acquire_multiplier_male"] = _p_acq_bounds()
    if fit_pdeath:
        lo = max(seed_pdeath / P_DEATH_BOUND_MULT, 1e-6)
        hi = min(seed_pdeath * P_DEATH_BOUND_MULT, 1.0)
        if hi <= lo:
            logger.warning(
                "%s: p_death bounds collapsed (seed=%g, mult=%g); skipping p_death fit",
                orig_disease_name,
                seed_pdeath,
                P_DEATH_BOUND_MULT,
            )
            fit_pdeath = False
        else:
            calib_pars["hc_p_death"] = dict(low=lo, high=hi, guess=seed_pdeath, log=True)
            logger.info(
                "%s: fitting p_death in [%g, %g] (seed=%g)",
                orig_disease_name,
                lo,
                hi,
                seed_pdeath,
            )
    if not fit_pdeath:
        logger.info(
            "%s: prevalence-only fit (use_death_objective=%s, nonmortal=%s, seed_pdeath=%g)",
            orig_disease_name,
            use_death_objective,
            is_nonmortal,
            seed_pdeath,
        )
    calib = ss.Calibration(
        sim=sim,
        calib_pars=calib_pars,
        build_fn=build_sim_local,
        eval_fn=eval_fn_local,
        eval_kw={"data": prev_df},
        total_trials=total_trials,
        n_workers=1,
        keep_db=False,
        die=True,
        reseed=False,
        sampler=optuna.samplers.TPESampler(seed=123),
    )
    calib.calibrate()
    diag_methods = OPTUNA_DIAGNOSTIC_METHODS_PDEATH if fit_pdeath else OPTUNA_DIAGNOSTIC_METHODS
    save_optuna_diagnostics(calib, orig_disease_name, results_dir, methods=diag_methods)
    calib.check_fit()
    best_fit_female = np.nan
    best_fit_male = np.nan
    best_fit_death_female = np.nan
    best_fit_death_male = np.nan
    best_fit_prev_norm = np.nan
    best_fit_death_norm = np.nan
    best_fit_total = np.nan
    if getattr(calib, "after_msim", None) is not None and getattr(calib.after_msim, "sims", None):
        (
            best_fit_female,
            best_fit_male,
            best_fit_death_female,
            best_fit_death_male,
            best_fit_prev_norm,
            best_fit_death_norm,
        ) = fit_by_sex_local(
            calib.after_msim.sims[0],
            prev_df,
            death_df if use_death_objective else None,
        )
        best_fit_total = float(WEIGHT_PREVALENCE * best_fit_prev_norm + WEIGHT_DEATH * best_fit_death_norm)
    save_calibration_fit_plot(calib, orig_disease_name, results_dir, prev_df, affected_sex=affected_sex)
    save_calibration_death_fit_plot(
        calib, orig_disease_name, results_dir, death_df, affected_sex=affected_sex
    )
    save_diagnostic_panel_from_pngs(orig_disease_name, results_dir)
    sc.saveobj(results_dir / f"calib_{orig_disease_name}_{sc.getdate()}.obj", calib)
    with open(results_dir / f"calibration_results_{orig_disease_name}.txt", "w") as f:
        f.write("HIV parameters:\n")
        f.write(f"hiv_beta_m2f: {HIV_BETA_M2F}\n")
        f.write(f"hiv_beta_m2c: {HIV_BETA_M2C}\n\n")
        f.write("Best parameters:\n")
        for k, v in calib.best_pars.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBest-fit sex-specific errors (prevalence raw MAE):\n")
        f.write(f"fit_female: {best_fit_female}\n")
        f.write(f"fit_male: {best_fit_male}\n")
        f.write(f"fit_death_female: {best_fit_death_female}\n")
        f.write(f"fit_death_male: {best_fit_death_male}\n")
        f.write(f"fit_prev_norm: {best_fit_prev_norm}\n")
        f.write(f"fit_death_norm: {best_fit_death_norm}\n")
        f.write(f"fit_total (weighted): {best_fit_total}\n")
        f.write(f"use_death_objective: {use_death_objective}\n")
        f.write(f"weight_prevalence: {WEIGHT_PREVALENCE}\n")
        f.write(f"weight_death: {WEIGHT_DEATH}\n")
        # Joint p_death calibration reporting
        best_pdeath_raw = calib.best_pars.get("hc_p_death", np.nan)
        try:
            best_pdeath_val = float(best_pdeath_raw)
        except (TypeError, ValueError):
            best_pdeath_val = np.nan
        ratio = (best_pdeath_val / seed_pdeath) if (pd.notna(best_pdeath_val) and seed_pdeath > 0) else np.nan
        f.write("\nJoint p_death calibration:\n")
        f.write(f"fit_pdeath: {fit_pdeath}\n")
        f.write(f"seed_p_death: {seed_pdeath}\n")
        f.write(f"best_p_death: {best_pdeath_val}\n")
        f.write(f"p_death_ratio (best/seed): {ratio}\n")
        if fit_pdeath and pd.notna(best_pdeath_val):
            lo_b = max(seed_pdeath / P_DEATH_BOUND_MULT, 1e-6)
            hi_b = min(seed_pdeath * P_DEATH_BOUND_MULT, 1.0)
            tol = 1e-3
            at_bound = (
                abs(np.log10(max(best_pdeath_val, 1e-12)) - np.log10(lo_b)) < tol
                or abs(np.log10(max(best_pdeath_val, 1e-12)) - np.log10(hi_b)) < tol
            )
            f.write(f"at_bound: {at_bound}\n")
            if at_bound:
                logger.warning(
                    "%s: best_p_death=%g hit p_death bound [%g, %g]; "
                    "likely data/mapping issue rather than a wider-prior need",
                    orig_disease_name,
                    best_pdeath_val,
                    lo_b,
                    hi_b,
                )
    best_f = calib.best_pars.get("hc_p_acquire_multiplier_female", np.nan)
    best_m = calib.best_pars.get("hc_p_acquire_multiplier_male", np.nan)
    if affected_sex == "female":
        p_legacy = best_f
    elif affected_sex == "male":
        p_legacy = best_m
    else:
        p_legacy = np.nanmean([best_f, best_m])
    logger.info(
        "Done: %s (affected_sex=%s) → p_acquire_female=%s, p_acquire_male=%s",
        orig_disease_name,
        affected_sex,
        f"{float(best_f):0.6f}" if pd.notna(best_f) else "NA",
        f"{float(best_m):0.6f}" if pd.notna(best_m) else "NA",
    )
    # Use the calibrated p_death when joint fit was active; otherwise fall back to seed.
    p_death_best_for_output = best_pdeath_val if (fit_pdeath and pd.notna(best_pdeath_val)) else seed_pdeath
    output_csv = results_dir / "calibrated_p_acquire.csv"
    row_df = pd.DataFrame(
        [
            {
                "condition": orig_disease_name,
                "affected_sex": affected_sex,
                "p_acquire": p_legacy,
                "p_acquire_female": best_f,
                "p_acquire_male": best_m,
                "p_death_seed": seed_pdeath,
                "p_death_best": p_death_best_for_output,
                "fit_pdeath": fit_pdeath,
                "fit_female": best_fit_female,
                "fit_male": best_fit_male,
                "fit_prev_norm": best_fit_prev_norm,
                "fit_death_norm": best_fit_death_norm,
                "fit_total": best_fit_total,
                "hiv_beta_m2f": HIV_BETA_M2F,
                "hiv_beta_m2c": HIV_BETA_M2C,
            }
        ]
    )
    if output_csv.exists():
        prev_out = pd.read_csv(output_csv)
        out = pd.concat([prev_out, row_df], ignore_index=True, sort=False)
        out = out.drop_duplicates(subset=["condition"], keep="last")
        out.to_csv(output_csv, index=False)
    else:
        row_df.to_csv(output_csv, index=False)

    # Also save a copy-paste-friendly aligned export in parameter-file condition order.
    aligned_csv = results_dir / "calibrated_p_acquire_aligned.csv"
    base = param_df[["condition"]].copy()
    latest = pd.read_csv(output_csv)
    aligned_cols = ["condition", "p_acquire_female", "p_acquire_male"]
    if "p_death_best" in latest.columns:
        aligned_cols.append("p_death_best")
    merged = base.merge(latest[aligned_cols], on="condition", how="left")
    if "p_death_best" in merged.columns:
        merged = merged.rename(columns={"p_death_best": "p_death"})
        # Fall back to seed p_death in the parameters CSV for conditions we didn't calibrate.
        seed_map = param_df.set_index("condition")["p_death"]
        merged["p_death"] = merged["p_death"].fillna(merged["condition"].map(seed_map))
    merged.to_csv(aligned_csv, index=False)
    return calib


def _parse_args():
    parser = argparse.ArgumentParser(description="Calibrate disease p_acquire with prevalence and death targets.")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Conditions to calibrate (default: module-level `conditions` list).",
    )
    parser.add_argument("--total-trials", type=int, default=None, help="Optuna trials per condition.")
    parser.add_argument("--smoke", action="store_true", help="Short run with 5 trials.")
    parser.add_argument("--weight-prev", type=float, default=None, help="Objective weight for prevalence.")
    parser.add_argument("--weight-death", type=float, default=None, help="Objective weight for death rates.")
    pdeath_group = parser.add_mutually_exclusive_group()
    pdeath_group.add_argument(
        "--fit-pdeath",
        dest="fit_pdeath",
        action="store_true",
        default=None,
        help="Include p_death in the Optuna search (joint fit). Default: on when death data present.",
    )
    pdeath_group.add_argument(
        "--no-fit-pdeath",
        dest="fit_pdeath",
        action="store_false",
        default=None,
        help="Prevalence-only fit; p_death stays at the seed value.",
    )
    parser.add_argument(
        "--pdeath-bound-mult",
        type=float,
        default=None,
        help="Multiplier for p_death bounds around the seed (default 10x).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Output directory for this run (created if missing). "
            "Reuse the same path across batch invocations to accumulate all conditions "
            "into one calibrated_p_acquire.csv."
        ),
    )
    return parser.parse_args()


# Main
if __name__ == "__main__":
    args = _parse_args()
    if args.smoke:
        total_trials = 5
    elif args.total_trials is not None:
        total_trials = int(args.total_trials)
    if args.weight_prev is not None:
        WEIGHT_PREVALENCE = float(args.weight_prev)
    if args.weight_death is not None:
        WEIGHT_DEATH = float(args.weight_death)
    if args.fit_pdeath is not None:
        FIT_PDEATH = bool(args.fit_pdeath)
    if args.pdeath_bound_mult is not None:
        P_DEATH_BOUND_MULT = float(args.pdeath_bound_mult)
    if args.results_dir is not None:
        results_dir = Path(args.results_dir).expanduser()
        if not results_dir.is_absolute():
            results_dir = REPO_ROOT / results_dir

    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results directory: %s", results_dir)
    run_conditions = args.conditions if args.conditions else conditions

    for disease_name in run_conditions:
        affected_sex = get_affected_sex(disease_name)
        if not has_observed_prevalence_data_for_affected_sex(disease_name, prev_df, affected_sex):
            continue
        DiseaseClass = try_get_condition_class(disease_name)
        if DiseaseClass is None:
            continue
        try:
            run_calibration(disease_name, DiseaseClass)
        except Exception as e:
            logger.exception("Error calibrating %s: %s", disease_name, e)
