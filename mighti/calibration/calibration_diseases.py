"""
Calibrate disease acquisition (p_acquire_multiplier) for specified conditions using
MIGHTI prevalence data. Uses disease classes from diseases_for_calibration (p_acquire=1
so the calibrated parameter is p_acquire_multiplier). Outputs best-fit parameter and
writes results to calibration_results_<Condition>.txt and calibrated_p_acquire.csv.
"""

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
from mighti.util.paths import get_data_dir

logger = logging.getLogger(__name__)

# Config
region = "eswatini"
init_year = 2007
end_year = 2023
total_trials = 100  # Set higher for production runs
HIV_BETA_M2F = 0.01688952663716571
HIV_BETA_M2C = 0.0444149203530297

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../MIGHTI/
DATA_DIR = get_data_dir()


def _resolve_data_file(filename):
    """
    Resolve a region input CSV within the repo.

    Preferred location is `data/processed/`. For developer/test workflows, we also
    allow falling back to `tests/test_data/`.
    """
    candidates = [
        DATA_DIR / filename,
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

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# Results under package dir so path works from repo root or from mighti/calibration/
results_dir = REPO_ROOT / "mighti" / "calibration" / "results" / f"calibration_{region}_{date_str}"
results_dir.mkdir(parents=True, exist_ok=True)

# Load prevalence once for eval
prev_df = pd.read_csv(path_prevalence)
param_df = pd.read_csv(path_parameters)
param_df["condition"] = param_df["condition"].astype(str).str.strip()
if "affected_sex" in param_df.columns:
    param_df["affected_sex"] = param_df["affected_sex"].astype(str).str.strip().str.lower()
else:
    param_df["affected_sex"] = "both"

# Conditions to calibrate (or use param_df['condition'].unique().tolist())
# conditions = ["ChromosomalAbnormalities","CongenitalHeartAnomalies","CongenitalMusculoskeletal","DiarrhealDiseases","DigestiveCongenitalAnomalies"]
all_conditions = param_df["condition"].dropna().unique().tolist()
# conditions = [
#     "AlzheimersDisease",
#     "BreastCancer",
#     "CardiovascularDiseases",
#     "CervicalCancer",
#     "ChronicKidneyDisease",
#     "ChronicLiverDisease",
#     "COPD",
#     "COVID19",
#     "DiarrhealDiseases",
# ]
conditions = all_conditions

OPTUNA_DIAGNOSTIC_METHODS = [
    "plot_optimization_history",
    "plot_param_importances",
    "plot_timeline",
    "plot_edf",
]

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

    sim = ss.Sim(
        dt=1,
        n_agents=10_000,
        total_pop=9_980_999,
        start=init_year,
        stop=end_year,
        diseases=[hiv, health_condition],
        networks=networks,
        demographics=[pregnancy, death],
        analyzers=[prevalence_analyzer],
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
            if "hc_" in k:
                hc.pars[k.replace("hc_", "")] = v
        return sim

    def fit_by_sex_local(sim, data):
        if isinstance(sim, ss.MultiSim):
            sim = sim.sims[0]
        female_fit = 0.0
        male_fit = 0.0
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

        for index, (age_low, _) in enumerate(prev_analyzer.age_bins):
            obs = data[data['Age'] == age_low][['Year', 'Age', f'{orig_disease_name}_female', f'{orig_disease_name}_male']].copy()

            # Ensure obs Year is int
            obs['Year'] = obs['Year'].astype(int)

            # Convert timevec (datetime) -> int year
            year_sim = pd.to_datetime(prev_analyzer.timevec).year

            sim_df = pd.DataFrame({
                'Year': year_sim.astype(int),
                'Age': age_low,
                'sim_female': prev_results[f'{disease_name.lower()}_prev_female_{index}'],
                'sim_male': prev_results[f'{disease_name.lower()}_prev_male_{index}']
            })
            merged = pd.merge(obs, sim_df, on=['Year', 'Age'], how='inner')
            if merged.empty:
                continue
            if affected_sex in {"female", "both"}:
                female_fit += (merged["sim_female"] - merged[f"{orig_disease_name}_female"]).abs().sum()
            if affected_sex in {"male", "both"}:
                male_fit += (merged["sim_male"] - merged[f"{orig_disease_name}_male"]).abs().sum()
        return float(female_fit), float(male_fit)

    def eval_fn_local(sim, data=None, **kwargs):
        female_fit, male_fit = fit_by_sex_local(sim, data)
        return float(female_fit + male_fit)

    sim = make_sim(orig_disease_name, DiseaseClass)
    if affected_sex == "female":
        calib_pars = {"hc_p_acquire_multiplier_female": dict(low=0.00001, high=0.10, guess=0.011)}
    elif affected_sex == "male":
        calib_pars = {"hc_p_acquire_multiplier_male": dict(low=0.00001, high=0.10, guess=0.011)}
    else:
        calib_pars = {
            "hc_p_acquire_multiplier_female": dict(low=0.00001, high=0.10, guess=0.011),
            "hc_p_acquire_multiplier_male": dict(low=0.00001, high=0.10, guess=0.011),
        }
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
    save_optuna_diagnostics(calib, orig_disease_name, results_dir)
    calib.check_fit()
    best_fit_female = np.nan
    best_fit_male = np.nan
    best_fit_total = np.nan
    if getattr(calib, "after_msim", None) is not None and getattr(calib.after_msim, "sims", None):
        best_fit_female, best_fit_male = fit_by_sex_local(calib.after_msim.sims[0], prev_df)
        best_fit_total = best_fit_female + best_fit_male
    save_calibration_fit_plot(calib, orig_disease_name, results_dir, prev_df, affected_sex=affected_sex)
    save_diagnostic_panel_from_pngs(orig_disease_name, results_dir)
    sc.saveobj(results_dir / f"calib_{orig_disease_name}_{sc.getdate()}.obj", calib)
    with open(results_dir / f"calibration_results_{orig_disease_name}.txt", "w") as f:
        f.write("HIV parameters:\n")
        f.write(f"hiv_beta_m2f: {HIV_BETA_M2F}\n")
        f.write(f"hiv_beta_m2c: {HIV_BETA_M2C}\n\n")
        f.write("Best parameters:\n")
        for k, v in calib.best_pars.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBest-fit sex-specific errors:\n")
        f.write(f"fit_female: {best_fit_female}\n")
        f.write(f"fit_male: {best_fit_male}\n")
        f.write(f"fit_total: {best_fit_total}\n")
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
    output_csv = results_dir / "calibrated_p_acquire.csv"
    row_df = pd.DataFrame(
        [
            {
                "condition": orig_disease_name,
                "affected_sex": affected_sex,
                "p_acquire": p_legacy,
                "p_acquire_female": best_f,
                "p_acquire_male": best_m,
                "fit_female": best_fit_female,
                "fit_male": best_fit_male,
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
    merged = base.merge(
        latest[["condition", "p_acquire_female", "p_acquire_male"]],
        on="condition",
        how="left",
    )
    merged.to_csv(aligned_csv, index=False)
    return calib


# Main
if __name__ == "__main__":
    for disease_name in conditions:
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
