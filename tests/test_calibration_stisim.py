import starsim as ss
import stisim as sti
import mighti as mi
import pandas as pd
import numpy as np
import sciris as sc
import pytest

n_agents = 10_000
inityear = 2007
endyear = 2010
country = "Eswatini"
disease = "Type2Diabetes"
diseases = ['HIV', disease]
csv_prevalence = "test_data/eswatini_prevalence.csv"
csv_path_age = '../mighti/data/eswatini_age_distribution_2007.csv'
csv_path_params = '../mighti/data/eswatini_parameters_dt.csv'

# Settings
debug = True  # If True, this will do smaller runs that can be run locally for debugging
do_save = True

def make_sim():
    
    ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))
    
    # Initialize networks
    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]
    
    prevalence_data_df = pd.read_csv(csv_prevalence)

    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases, prevalence_data=prevalence_data_df, inityear=inityear
    )

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(
            disease, prevalence_data, age_bins, sim, size)

    init_prev = ss.bernoulli(get_prevalence_function(disease))
    t2d = mi.Type2Diabetes(csv_path=csv_path_params, pars={"init_prev": init_prev})
    hiv = sti.HIV(beta_m2f=0.05, beta_m2c=0.025, init_prev=0.15)

    sim = ss.Sim(
        dt=1/12,
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        diseases=[t2d, hiv],
        copy_inputs=False,
        verbose=0,
    )
    
    sim.init()
    
    return sim

def build_sim(sim, calib_pars=None, **kwargs):
    reps = kwargs.get('n_reps', 1)

    hiv = sim.diseases.hiv
    t2d = sim.diseases.type2diabetes

    for k, pars in calib_pars.items():
        v = pars['value']
        if 'hiv_' in k:
            k = k.replace('hiv_', '')
            hiv.pars[k] = v
        elif 'type2diabetes_' in k:
            k = k.replace('type2diabetes_', '')
            t2d.pars[k] = v
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')
    return sim


def run_calib(calib_pars=None):
    sc.heading('Beginning calibration')

    # Make the sim and data
    sim = make_sim()
    data = pd.read_csv('test_data/eswatini_calib.csv')

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
    calib.check_fit()
    calib.plot_optuna('plot_param_importances')

    sc.printcyan('\nShrinking calibration...')
    cal = calib.shrink()

    return sim, calib, cal


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    # Define the calibration parameters
    calib_pars = dict(
        hiv_beta_m2f = dict(low=0.01, high=0.10, guess=0.05),
        type2diabetes_p_acquire = dict(low=0.00001, high=0.1, guess=0.001, suggest_type='suggest_float', log=True),
      )

    sim, calib, cal = run_calib(calib_pars=calib_pars)

    sc.toc(T)
    print('Done.')