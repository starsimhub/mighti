"""
Test calibration
"""

#%% Imports and settings
import starsim as ss
import sciris as sc
import stisim as sti
import pandas as pd
import mighti as mi

do_plot = 1
do_save = 0
n_agents = 2e3

# Settings
debug = True  # If True, this will do smaller runs that can be run locally for debugging
do_save = True

def ret(*values):
    return values if __name__ == 'main' else None

n_agents = 10_000
inityear = 2007
endyear = 2011
region = 'eswatini'

csv_prevalence = f'test_data/{region}_hiv.csv'
csv_path_age = f'../mighti/data/{region}_age_distribution_{inityear}.csv'
csv_path_fertility = f'../mighti/data/{region}_asfr.csv'
csv_path_death = f'../mighti/data/{region}_mortality_rates_{inityear}.csv'


import prepare_data_for_year
prepare_data_for_year.prepare_data_for_year(region,inityear)

def make_sim():

    prevalence_data_df = pd.read_csv(csv_prevalence)

    prevalence_data, age_bins = mi.initialize_prevalence_data(
        ['HIV'], prevalence_data=prevalence_data_df, inityear=inityear
    )
    
    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(
            disease, prevalence_data, age_bins, sim, size)
    
    hiv = sti.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), init_prev_data=None,
                  beta_m2f=0.05, beta_m2c=0.025)

    mortality_rates = pd.read_csv(csv_path_death)
    if 'age' in mortality_rates.columns:
        mortality_rates = mortality_rates.rename(columns={'age': 'AgeGrpStart'})
    elif 'Age' in mortality_rates.columns:
        mortality_rates = mortality_rates.rename(columns={'Age': 'AgeGrpStart'})
        
    death_rates = {'death_rate': mortality_rates, 'rate_units': 1}
    death = ss.Deaths(death_rates)  # Use Demographics class implemented in mighti
    fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    
    ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))
    
    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]

    sim = ss.Sim(
        n_agents=n_agents,
        # total_pop=9980999,
        start=inityear,
        stop=endyear,
        people = ppl,
        demographics=[pregnancy, death],
        diseases=[hiv],
        networks=networks,
    )

    sim.init()

    return sim


def build_sim(sim, calib_pars):

    hiv = sim.diseases.hiv

    # Apply the calibration parameters
    for k, pars in calib_pars.items():  # Loop over the calibration parameters

        v = pars['value']
        if 'hiv_' in k:  # HIV parameters
            k = k.replace('hiv_', '')  # Strip off identifying part of parameter name
            hiv.pars[k] = v
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    return sim


def run_calib(calib_pars=None):
    sc.heading('Beginning calibration')

    # Make the sim and data
    sim = make_sim()
    data = pd.read_csv(f'test_data/{region}_calib.csv')

    # Make the calibration
    calib = sti.Calibration(
        sim=sim,
        calib_pars=calib_pars,
        build_fn=build_sim,
        total_trials=2,
        n_workers=1,
        die=True,
        reseed=False,
        debug=debug,
        data=data,
        save_results=True,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()
    
    print(type(sim), hasattr(sim, "results"))
    print(sim.results)

    calib.check_fit()
    calib.plot_optuna('plot_param_importances')

    sc.printcyan('\nShrinking calibration...')
    cal = calib.shrink()

    return sim, calib, cal

    # Run as a script
if __name__ == '__main__':

    T = sc.tic()

    # Define the calibration parameters
    calib_pars = dict(
        hiv_beta_m2f = dict(low=0.01, high=0.10, guess=0.05),
    )

    sim, calib, cal = run_calib(calib_pars=calib_pars)

    sc.toc(T)
    print('Done.')



