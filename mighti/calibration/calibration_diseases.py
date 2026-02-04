# """
# NOTICE; THIS DOES NOT WORK WITH STARSIM==2.3.1, starsim==2.2.0, stisim==1.0.1, numpy==2.2.6
# Calibrate disease acquisition parameter (p_acquire) for a specified condition
# using MIGHTI and prevalence data. Outputs best-fit parameter and comparison
# of observed vs. simulated prevalence by age and sex.
# """
"""
NOTICE; THIS DOES NOT WORK WITH STARSIM==2.3.1, starsim==2.2.0, stisim==1.0.1, numpy==2.2.6
Calibrate disease acquisition parameter (p_acquire) for a specified condition
using MIGHTI and prevalence data. Outputs best-fit parameter and comparison
of observed vs. simulated prevalence by age and sex.
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
region = 'eswatini'
init_year = 1990
end_year = 2023
total_trials = 100  # Set higher for production runs

# Resolve package-relative data paths (no hard-coded local paths)
BASE_MIGHTI_DIR = Path(__file__).resolve().parents[1]  # .../mighti/
DATA_DIR = BASE_MIGHTI_DIR / "data"

# Paths
path_prevalence = str(DATA_DIR / f"{region}_prevalence.csv")
path_parameters = str(DATA_DIR / f"{region}_parameters.csv")
date_str = datetime.now().strftime("%Y%m%d")
results_dir = Path("outputs") / f"calibration_{region}_{date_str}"
results_dir.mkdir(parents=True, exist_ok=True)

# Load prevalence and parameter data
prev_df = pd.read_csv(path_prevalence)
param_df = pd.read_csv(path_parameters)
# conditions = param_df['condition'].unique().tolist()

conditions = ['CardiovascularDiseases']
# conditions = ['LungCancer', 'ProstateCancer',
#               'AlcoholUseDisorder', 'RoadInjuries', 'ChronicLiverDisease',
#               'Asthma']

# Try importing condition class dynamically
def try_get_condition_class(name):
    try:
        module = import_module("mighti.calibration.diseases_for_calibration")
        return getattr(module, name)
    except AttributeError:
        logger.warning("Skipping %s: not implemented in diseases_for_calibration.py", name)
        return None

def make_sim(disease_name, DiseaseClass):
    """
    Lightweight MIGHTI calibration sim — updated for StarSim 3.x
    Mirrors MIGHTI main structure but keeps the compact format.
    """

    # --- Prevalence setup ---
    diseases = ["HIV"] + conditions
    prevalence_data_df = pd.read_csv(f"mighti/data/{region}_prevalence.csv")
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases=diseases, prevalence_data=prevalence_data_df, inityear=init_year
    )

    def get_prevalence_function(disease):
        def prevalence_func(sim, uids, size=None):
            return mi.age_sex_dependent_prevalence(
                disease=disease, prevalence_data=prevalence_data,
                age_bins=age_bins, sim=sim, uids=uids,
            )
        return prevalence_func

    # -----------------------------------------------------------------
    # Basic configuration
    # -----------------------------------------------------------------
    disease_objects = []

    hiv = sti.HIV()  

    # Assign prevalence
    prev_func = get_prevalence_function('HIV')
    hiv.pars.init_prev = ss.bernoulli(
        p=lambda sim, uids, size=None: prev_func(sim, uids, size)
    )

    # Transmission parameters
    hiv.pars.beta = {
        'structuredsexual': [0.029594299274445842, 0.029594299274445842],
        'maternal': [0.0011249414706988527, 0.0011249414706988527],
    }
    hiv.pars.include_aids_deaths = True
    hiv.pars.p_hiv_death = ss.bernoulli(p=0.00015)
    hiv.pars.include_care = True
    hiv.pars.art_efficacy = 0.9

    disease_objects.append(hiv)

    # -----------------------------------------------------------------
    # Disease under calibration
    # -----------------------------------------------------------------
    def make_init_prev_func(disease):
        prev_func = get_prevalence_function(disease)
        return lambda sim, uids, size=None: prev_func(sim, uids, size)

    for disease in conditions:
        disease_class = getattr(mi, disease, None)
        if disease_class:
            init_prev = ss.bernoulli(p=make_init_prev_func(disease))
            disease_obj = disease_class(csv_path=path_parameters, pars={"init_prev": init_prev})
            disease_objects.append(disease_obj)

    # -----------------------------------------------------------------
    # Demographics
    # -----------------------------------------------------------------
    death_rates = {"death_rate": pd.read_csv(f"mighti/data/{region}_mortality_rates.csv"), "rate_units": 1}
    death = ss.Deaths(pars=death_rates)

    fertility_rate = {"fertility_rate": pd.read_csv(f"mighti/data/{region}_asfr.csv")}
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]

    # -----------------------------------------------------------------
    # Analyzer
    # -----------------------------------------------------------------
    prevalence_analyzer = mi.PrevalenceAnalyzer(
        prevalence_data=prevalence_data, diseases=diseases
    )

    # -----------------------------------------------------------------
    # Build simulation
    # -----------------------------------------------------------------
    sim = ss.Sim(
        dt=1,
        n_agents=10_000,
        total_pop=9_980_999,
        start=init_year,
        stop=end_year,
        diseases=disease_objects,
        networks=networks,
        demographics=[pregnancy, death],
        analyzers=[prevalence_analyzer],
        copy_inputs=False,
        label=f"Calibration - {disease_name}",
    )

    sim.init()
    return sim


# Build function
def build_sim(sim, calib_pars):
    hc = sim.diseases[disease_name.lower()]
    for k, pars in calib_pars.items():
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue
        v = pars['value']
        if 'hc_' in k:
            k = k.replace('hc_', '')
            hc.pars[k] = v
    return sim


def eval_fn(sim, data=None, sim_result_list=None, weights=None, df_res_list=None):
    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]
    fit = 0

    # Find prevalence analyzer dynamically
    for label, analyzer in sim.analyzers.items():
        if isinstance(analyzer, mi.PrevalenceAnalyzer):
            prev_analyzer = analyzer
            prev_results = sim.results[label]
            break
    else:
        raise KeyError("No PrevalenceAnalyzer found in sim.analyzers")

    key_base = disease_name.lower()     
    disease = sim.diseases[key_base]
    sex = disease.pars.affected_sex.lower()

    for index, (age_low, age_high) in enumerate(prev_analyzer.age_bins):
        obs = data[data['Age'] == age_low][['Year', 'Age']]
        sim_df = pd.DataFrame({'Year': prev_analyzer.timevec, 'Age': age_low})

        if sex in ['female', 'both']:
            obs[f'{disease_name}_female'] = data[data['Age'] == age_low][f'{disease_name}_female'].values
            sim_df['sim_female'] = prev_results[f'{key_base}_prev_female_{index}']   # changed
            sim_df['error_f'] = abs(sim_df['sim_female'] - obs[f'{disease_name}_female'])

        if sex in ['male', 'both']:
            obs[f'{disease_name}_male'] = data[data['Age'] == age_low][f'{disease_name}_male'].values
            sim_df['sim_male'] = prev_results[f'{key_base}_prev_male_{index}']       # changed
            sim_df['error_m'] = abs(sim_df['sim_male'] - obs[f'{disease_name}_male'])
       
        # Merge and sum
        obs["Year"] = pd.to_datetime(obs["Year"], errors="coerce").dt.year.astype("Int64")
        sim_df["Year"] = pd.to_datetime(sim_df["Year"], errors="coerce").dt.year.astype("Int64")
        merged = pd.merge(obs, sim_df, on=['Year', 'Age'], how='inner')
 
        if 'error_f' in merged:
            fit += merged['error_f'].sum()
        if 'error_m' in merged:
            fit += merged['error_m'].sum()

    return fit


# Calibration runner
def run_calibration(disease_name, DiseaseClass):
    sim = make_sim(disease_name, DiseaseClass)

    calib_pars = dict(
        hc_p_acquire_multiplier=dict(low=0.0001, high=0.10, guess=0.01),
    )

    calib = ss.Calibration(
        sim=sim,
        calib_pars=calib_pars,
        build_fn=build_sim,
        eval_fn=eval_fn,
        eval_kw={'data': prev_df},
        total_trials=total_trials,
        n_workers=1,
        keep_db=False,
        die=True,
        reseed=False,
        sampler=optuna.samplers.TPESampler(seed=123),
    )
    calib.calibrate()
    calib.check_fit()

    sc.saveobj(f'{results_dir}/calib_{disease_name}_{sc.getdate()}.obj', calib)

    with open(f'{results_dir}/calibration_results_{disease_name}.txt', 'w') as f:
        f.write('Best parameters:\n')
        for k, v in calib.best_pars.items():
            f.write(f'{k}: {v}\n')

    logger.info(
        "Done: %s → best p_acquire = %0.4f",
        disease_name,
        float(calib.best_pars["hc_p_acquire_multiplier"]),
    )
    
    # Save to a single CSV file with all calibrated p_acquire values
    output_csv = f'{results_dir}/calibrated_p_acquire.csv'
    row = {'condition': disease_name, 'p_acquire': calib.best_pars['hc_p_acquire_multiplier']}
    row_df = pd.DataFrame([row])
    
    if not os.path.exists(output_csv):
        row_df.to_csv(output_csv, index=False)
    else:
        row_df.to_csv(output_csv, mode='a', header=False, index=False)
    
    return calib


# Main loop
if __name__ == '__main__':
    for disease_name in conditions:
        DiseaseClass = try_get_condition_class(disease_name)
        if DiseaseClass is not None:
            try:
                run_calibration(disease_name, DiseaseClass)
            except Exception as e:
                logger.exception("Error calibrating %s: %s", disease_name, e)

