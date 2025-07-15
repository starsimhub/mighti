"""
Calibrate disease acquisition parameter (p_acquire) for a specified condition
using MIGHTI and prevalence data. Outputs best-fit parameter and comparison
of observed vs. simulated prevalence by age and sex.
"""


import optuna
import mighti as mi
import pandas as pd
import sciris as sc
import starsim as ss
import stisim as sti


# Set the name of the disease to calibrate
from mighti.calibration.diseases_for_calibration import Type2Diabetes as DiseaseClass  
disease_name = 'Type2Diabetes'  

# Set the starting year for calibration
init_year = 2007                
total_trials = 10   # Use a small number for testing; increase to 100+ for full calibration

path_prevalence = '../data/eswatini_prevalence.csv'
path_parameters = '../data/eswatini_parameters.csv'


def make_sim():
    # Best pars: {'hiv_beta_m2f': 0.011023883426646121, 'hiv_beta_m2c': 0.044227226248848076} seed: 12345
    hiv = sti.HIV(beta_m2f=0.011023883426646121, beta_m2c=0.044227226248848076, init_prev=0.15)
    
    # Dynamically select disease constructor
    health_condition_cls = DiseaseClass

    prev_data = pd.read_csv(path_prevalence)
    prev_data, age_bins = mi.initialize_prevalence_data([disease_name], prev_data, init_year)  # 2007 = init_year
    
    def get_prev_fn(disease):
        return lambda mod, sim, size: mi.age_sex_dependent_prevalence(disease, prev_data, age_bins, sim, size)
    
    health_condition = health_condition_cls(
        pars={'init_prev': ss.bernoulli(get_prev_fn(disease_name))},
        csv_path=path_parameters
    )
    
    fertility_rate = {'fertility_rate': pd.read_csv('../data/eswatini_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    death_rates = {'death_rate': pd.read_csv('../data/eswatini_mortality_rates.csv'), 'rate_units': 1}
    death = ss.Deaths(death_rates)  

    sexual = sti.StructuredSexual()
    maternal = ss.MaternalNet()

    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=pd.read_csv(path_prevalence), diseases=['HIV', disease_name])

    sim = ss.Sim(
        dt=1,
        unit='month',
        n_agents=10000,
        total_pop=9980999,
        start=init_year,
        stop=2023,
        diseases=[hiv, health_condition],
        networks=[sexual, maternal],
        demographics=[pregnancy, death],
        analyzers=[prevalence_analyzer],
    )

    sim.init()
    return sim


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
        else:
            raise NotImplementedError(f'Parameter {k} not recognized in build_sim()')

    return sim


def eval_fn(sim, data=None, sim_result_list=None, weights=None, df_res_list=None):

    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]

    fit = 0
    prev_analyzer = sim.analyzers.prevalence_analyzer
    prev_results = sim.results.prevalence_analyzer

    # Health condition prevalence
    for index, (age_low, age_high) in enumerate(prev_analyzer.age_bins):
        obs = data[data['Age'] == age_low][['Year', 'Age', f'{disease_name}_female', f'{disease_name}_male']]
        sim_df = pd.DataFrame({
            'Year': prev_analyzer.timevec,
            'Age': age_low,
            'sim_female': prev_results[f'{disease_name}_prev_female_{index}'],
            'sim_male': prev_results[f'{disease_name}_prev_male_{index}']
        })
        merged = pd.merge(obs, sim_df, on=['Year', 'Age'], how='inner')
        merged['error'] = abs(merged['sim_female'] - merged[f'{disease_name}_female']) + abs(merged['sim_male'] - merged[f'{disease_name}_male'])
        fit += merged['error'].sum()

    return fit


def run_calib(calib_pars=None, total_trials=10, keep_db=False):
    sim = make_sim()
    data = pd.read_csv(path_prevalence)

    calib = ss.Calibration(
        sim=sim,
        calib_pars=calib_pars,
        build_fn=build_sim,
        eval_fn=eval_fn,
        eval_kw={'data': data},
        total_trials=total_trials,
        n_workers=1,
        keep_db=keep_db,
        die=True,
        reseed=False,
        sampler=optuna.samplers.TPESampler(seed=12345) 
    )
    calib.calibrate()
    calib.check_fit()

    # Best-fit summary
    best_val = calib.best_pars['hc_p_acquire_multiplier']
    print(f'\nBest-fit p_acquire for {disease_name} = {best_val:.3f}\n')
    
    return calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    # Define the calibration parameters for health condition
    calib_pars = dict(
        hc_p_acquire_multiplier = dict(low=0.0001, high=0.10, guess=0.011),
    )

    calib = run_calib(calib_pars=calib_pars, total_trials=total_trials, keep_db=False)

    sc.toc(T)
    print('Done.')
    