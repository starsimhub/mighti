# """
# NOTICE; THIS DOES NOT WORK WITH STARSIM==2.3.1, starsim==2.2.0, stisim==1.0.1, numpy==2.2.6
# Calibrate disease acquisition parameter (p_acquire) for a specified condition
# using MIGHTI and prevalence data. Outputs best-fit parameter and comparison
# of observed vs. simulated prevalence by age and sex.
# """






import os
import optuna
import mighti as mi
import pandas as pd
import sciris as sc
import starsim as ss
import stisim as sti
from importlib import import_module

# Config
region = 'eswatini'
init_year = 1990
end_year = 2023
total_trials = 100  # Set higher for production runs

# base_path = "../"
base_path = "/Users/yamamn02/Documents/MIGHTI/mighti/"
# Paths
path_prevalence = f"{base_path}data/{region}_prevalence.csv"
path_parameters = f"{base_path}data/eswatini_parameters.csv"
results_dir = f'results/calibration_{region}_1008'
os.makedirs(results_dir, exist_ok=True)

# Load prevalence and parameter data
prev_df = pd.read_csv(path_prevalence)
param_df = pd.read_csv(path_parameters)
# conditions = param_df['condition'].unique().tolist()

# conditions = ['RoadInjuries']
conditions = ['LungCancer', 'ProstateCancer',
              'AlcoholUseDisorder', 'RoadInjuries', 'ChronicLiverDisease',
              'Asthma']

# Try importing condition class dynamically
def try_get_condition_class(name):
    try:
        module = import_module("mighti.calibration.diseases_for_calibration")
        return getattr(module, name)
    except AttributeError:
        print(f"⚠️ Skipping {name}: not implemented in diseases_for_calibration.py")
        return None

# Build sim
def make_sim(disease_name, DiseaseClass):
    # Best pars: {'hiv_beta_m2f': 0.029594299274445842, 'hiv_beta_m2c': 0.0011249414706988527}

    hiv = sti.HIV(beta_m2f= 0.029594299274445842, beta_m2c=0.0011249414706988527, init_prev=0.15)
    prev_data, age_bins = mi.initialize_prevalence_data([disease_name], prev_df, init_year)

    def get_prev_fn(disease):
        return lambda mod, sim, size: mi.age_sex_dependent_prevalence(disease, prev_data, age_bins, sim, size)

    health_condition = DiseaseClass(
        pars={'init_prev': ss.bernoulli(get_prev_fn(disease_name))},
        csv_path=path_parameters
    )

    fertility = {'fertility_rate': pd.read_csv(f"{base_path}data/{region}_asfr.csv")}
    pregnancy = ss.Pregnancy(pars=fertility)
    death = ss.Deaths({'death_rate': pd.read_csv(f"{base_path}data/{region}_mortality_rates.csv"), 'rate_units': 1})
    
    sexual = sti.StructuredSexual()
    maternal = ss.MaternalNet()

    prevalence_analyzer = mi.PrevalenceAnalyzer_HIV(prevalence_data=prev_df, diseases=['HIV', disease_name])

    sim = ss.Sim(
        dt=1, 
        unit='year', 
        n_agents=10000, 
        total_pop=9980999,
        start=init_year, 
        stop=end_year,
        diseases=[hiv, health_condition],
        networks=[sexual, maternal],
        demographics=[pregnancy, death],
        analyzers=[prevalence_analyzer],
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


# Eval function
def eval_fn(sim, data=None, sim_result_list=None, weights=None, df_res_list=None):
    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]
    fit = 0
    prev_analyzer = sim.analyzers.prevalence_analyzer
    prev_results = sim.results.prevalence_analyzer

    # Get affected sex for this disease
    disease = sim.diseases[disease_name.lower()]
    sex = disease.pars.affected_sex.lower()  # 'male', 'female', or 'both'

    for index, (age_low, age_high) in enumerate(prev_analyzer.age_bins):
        obs = data[data['Age'] == age_low][['Year', 'Age']]
        sim_df = pd.DataFrame({'Year': prev_analyzer.timevec, 'Age': age_low})

        if sex in ['female', 'both']:
            obs[f'{disease_name}_female'] = data[data['Age'] == age_low][f'{disease_name}_female'].values
            sim_df['sim_female'] = prev_results[f'{disease_name}_prev_female_{index}']
            sim_df['error_f'] = abs(sim_df['sim_female'] - obs[f'{disease_name}_female'])

        if sex in ['male', 'both']:
            obs[f'{disease_name}_male'] = data[data['Age'] == age_low][f'{disease_name}_male'].values
            sim_df['sim_male'] = prev_results[f'{disease_name}_prev_male_{index}']
            sim_df['error_m'] = abs(sim_df['sim_male'] - obs[f'{disease_name}_male'])

        # Merge and sum
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

    print(f" Done: {disease_name} → best p_acquire = {calib.best_pars['hc_p_acquire_multiplier']:.4f}")
    
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
                print(f" Error calibrating {disease_name}: {e}")

