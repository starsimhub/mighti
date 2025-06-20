"""
Calibrate disease acquisition parameter (p_acquire) for a specified condition
using MIGHTI and prevalence data. Outputs best-fit parameter and comparison
of observed vs. simulated prevalence by age and sex.
"""


import sciris as sc
import starsim as ss
import stisim as sti
import mighti as mi
import pandas as pd


disease_name = 'Type2Diabetes'  # Set the name of the disease to calibrate
init_year = 2007                # Set the starting year for calibration
total_trials = 100              # Use a small number for testing; increase to 100+ for full calibration

path_prevalence = 'mighti/data/eswatini_prevalence.csv'
path_parameters = 'mighti/data/eswatini_parameters_gbd.csv'


def make_sim():
    hiv = sti.HIV(beta_m2f=0.05, beta_m2c=0.025, init_prev=0.15)
    
    # Dynamically select disease constructor
    health_condition_cls = getattr(mi, disease_name)

    prev_data = pd.read_csv(path_prevalence)
    prev_data, age_bins = mi.initialize_prevalence_data([disease_name], prev_data, init_year)  # 2007 = init_year
    
    def get_prev_fn(disease):
        return lambda mod, sim, size: mi.age_sex_dependent_prevalence(disease, prev_data, age_bins, sim, size)
    
    health_condition = health_condition_cls(
        pars={'init_prev': ss.bernoulli(get_prev_fn(disease_name))},
        csv_path=path_parameters
    )
    
    fertility_rate = {'fertility_rate': pd.read_csv('mighti/data/eswatini_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    death_rates = {'death_rate': pd.read_csv('mighti/data/eswatini_mortality_rates_2007.csv'), 'rate_units': 1}
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

    for k, v in calib_pars.items():
        if k == 'rand_seed':
            sim.pars.rand_seed = v
            continue

        if 'hc_' in k: 
            param_name = k.replace('hc_', '')
            hc.pars[param_name] = v
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
    )
    calib.calibrate()
    calib.check_fit()

    # Best-fit summary
    best_val = calib.best_pars['hc_p_acquire_multiplier']
    print(f'\nBest-fit p_acquire for {disease_name} = {best_val:.3f}\n')
    
    # Set the estimated p_acquire
    sim.diseases[disease_name.lower()].pars['p_acquire'] = best_val
    
    # Reinitialize and run for 1 time step to get baseline stats
    sim.run()
    
    # Get analyzer results
    analyzer = sim.analyzers.prevalence_analyzer
    res = sim.results.prevalence_analyzer
    
    print(f"\nInitial Prevalence Comparison by Age Bin — {disease_name}")
    print(f"{'Age':>5} | {'Obs F':>7} | {'Sim F':>7} | {'Obs M':>7} | {'Sim M':>7}")
    print("-" * 42)
    
    for i, (age_low, age_high) in enumerate(sim.analyzers.prevalence_analyzer.age_bins):
        # Simulated
        sim_f = res[f'{disease_name}_prev_female_{i}'][0]
        sim_m = res[f'{disease_name}_prev_male_{i}'][0]
    
        # Observed
        obs_row = data[(data['Year'] == init_year) & (data['Age'] == age_low)]
        if obs_row.empty:
            continue  # skip if no observed data
    
        obs_f = obs_row[f'{disease_name}_female'].values[0]
        obs_m = obs_row[f'{disease_name}_male'].values[0]
    
        print(f"{age_low:>5} | {obs_f:7.2%} | {sim_f:7.2%} | {obs_m:7.2%} | {sim_m:7.2%}")
    return calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    # Define the calibration parameters for health condition
    calib_pars = dict(
        hc_p_acquire_multiplier = dict(low=0.01, high=0.10, guess=0.05),
    )

    calib = run_calib(calib_pars=calib_pars, total_trials=total_trials, keep_db=False)

    sc.toc(T)
    print('Done.')
    