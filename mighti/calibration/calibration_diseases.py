"""
Calibrate disease acquisition (p_acquire_multiplier) for specified conditions using
MIGHTI prevalence data. Uses disease classes from diseases_for_calibration (p_acquire=1
so the calibrated parameter is p_acquire_multiplier). Outputs best-fit parameter and
writes results to calibration_results_<Condition>.txt and calibrated_p_acquire.csv.
"""

import os
from pathlib import Path
import optuna
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
total_trials = 100  # Set higher for production runs

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../MIGHTI/
DATA_DIR = REPO_ROOT / "data" / "processed"


def _resolve_data_file(filename: str) -> Path:
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
path_parameters = str(_resolve_data_file(f"{region}_parameters.csv"))
path_fertility = str(_resolve_data_file(f"{region}_asfr.csv"))
path_mortality = str(_resolve_data_file(f"{region}_mortality_rates.csv"))

date_str = datetime.now().strftime("%Y%m%d")
# Results under package dir so path works from repo root or from mighti/calibration/
results_dir = REPO_ROOT / "mighti" / "calibration" / "results" / f"calibration_{region}_{date_str}"
results_dir.mkdir(parents=True, exist_ok=True)

# Load prevalence once for eval
prev_df = pd.read_csv(path_prevalence)
param_df = pd.read_csv(path_parameters)

# conditions = param_df['condition'].unique().tolist()
# Conditions to calibrate (or use param_df['condition'].unique().tolist())
conditions = ["AcuteHepatitis"]


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
    prevalence_data_df = pd.read_csv(path_prevalence)
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases=diseases,
        prevalence_data=prevalence_data_df,
        inityear=init_year,
    )

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

    # HIV
    hiv = sti.HIV()
    hiv.pars.init_prev = ss.bernoulli(
        p=lambda sim, uids, size=None: get_prevalence_function("HIV")(sim, uids, size)
    )
    hiv.pars.beta = {
        "structuredsexual": [0.029594299274445842, 0.029594299274445842],
        "maternal": [0.0011249414706988527, 0.0011249414706988527],
    }
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

    def eval_fn_local(sim, data=None, **kwargs):
        if isinstance(sim, ss.MultiSim):
            sim = sim.sims[0]
        fit = 0.0
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
        key_base = orig_disease_name.lower()

        for index, (age_low, _) in enumerate(prev_analyzer.age_bins):
            obs = data[data['Age'] == age_low][['Year', 'Age', f'{disease_name}_female', f'{disease_name}_male']].copy()

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
            fit += (merged["sim_female"] - merged[f"{orig_disease_name}_female"]).abs().sum()
            fit += (merged["sim_male"] - merged[f"{orig_disease_name}_male"]).abs().sum()
        return float(fit)

    sim = make_sim(orig_disease_name, DiseaseClass)
    calib_pars = {"hc_p_acquire_multiplier": dict(low=0.0001, high=0.10, guess=0.011)}
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
    calib.check_fit()
    sc.saveobj(results_dir / f"calib_{orig_disease_name}_{sc.getdate()}.obj", calib)
    with open(results_dir / f"calibration_results_{orig_disease_name}.txt", "w") as f:
        f.write("Best parameters:\n")
        for k, v in calib.best_pars.items():
            f.write(f"{k}: {v}\n")
    logger.info("Done: %s → best p_acquire_multiplier = %0.6f", orig_disease_name, float(calib.best_pars["hc_p_acquire_multiplier"]))
    output_csv = results_dir / "calibrated_p_acquire.csv"
    row_df = pd.DataFrame([{"condition": orig_disease_name, "p_acquire": calib.best_pars["hc_p_acquire_multiplier"]}])
    if not output_csv.exists():
        row_df.to_csv(output_csv, index=False)
    else:
        row_df.to_csv(output_csv, mode="a", header=False, index=False)
    return calib


# Main
if __name__ == "__main__":
    for disease_name in conditions:
        DiseaseClass = try_get_condition_class(disease_name)
        if DiseaseClass is None:
            continue
        try:
            run_calibration(disease_name, DiseaseClass)
        except Exception as e:
            logger.exception("Error calibrating %s: %s", disease_name, e)
