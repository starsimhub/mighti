import sciris as sc
import starsim as ss
import stisim as sti
import mighti as mi
import pandas as pd


def make_sim():
    # fertility_rates = pd.read_csv('mighti/data/eswatini_asfr.csv')
    # death_rates = pd.read_csv('mighti/data/eswatini_mortality_rates_2007.csv') # TODO update to use yearly mortality rate data if desired

    hiv = sti.HIV(beta_m2f=0.05, beta_m2c=0.025, init_prev=0.15)
    fertility_rate = {'fertility_rate': pd.read_csv('mighti/data/eswatini_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    death_rates = {'death_rate': pd.read_csv('mighti/data/eswatini_mortality_rates_2007.csv'), 'rate_units': 1}
    death = ss.Deaths(death_rates)  # Assuming death_rate is a yearly rate

    sexual = sti.StructuredSexual()
    maternal = ss.MaternalNet()

    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=pd.read_csv('mighti/data/eswatini_prevalence.csv'), diseases=['HIV'])

    sim = ss.Sim(
        dt=1,
        unit = 'month',
        n_agents=10000,
        total_pop=9980999,
        start=1990,
        stop= 2023,
        diseases=hiv,
        networks=[sexual, maternal],
        demographics=[pregnancy, death],
        analyzers=prevalence_analyzer,
    )

    sim.init()

    return sim

def build_sim(sim, calib_pars):
    hiv = sim.diseases.hiv
    nw = sim.networks.structuredsexual

    # Apply the calibration parameters
    for k, pars in calib_pars.items():  # Loop over the calibration parameters
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue

        v = pars['value']
        if 'hiv_' in k:  # HIV parameters
            k = k.replace('hiv_', '')  # Strip off identifying part of parameter name
            hiv.pars[k] = v
        elif 'nw_' in k:  # Network parameters
            k = k.replace('nw_', '')  # As above
            if 'pair_form' in k:
                nw.pars[k].set(v)
            else:
                nw.pars[k] = v
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    return sim


def run_calib(calib_pars=None, total_trials=10, keep_db=False):
    """
    Run the calibration simulation with the given parameters.

    Args:
        calib_pars (dict): Dictionary of calibration parameters.
        total_trials (int): Total number of trials for the calibration.
        keep_db (bool): Whether to keep the database after calibration. If kept it can be used to continue a calibration with more trials
    """
    sim = make_sim()
    #
    # if calib_pars is not None:
    #     sim = build_sim(sim, calib_pars)

    data = pd.read_csv('mighti/data/eswatini_prevalence.csv')

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

    # Return the results for further analysis
    return calib



def eval_fn(sim, data=None, sim_result_list=None, weights=None, df_res_list=None):
    """
    Custom evaluation function for HIV calibration
    """
    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]

    fit = 0
    prev_analyzer = sim.analyzers.prevalence_analyzer
    prev_results = sim.results.prevalence_analyzer

    # HIV prevalence
    for index, (age_low, age_high) in enumerate(sim.analyzers.prevalence_analyzer.age_bins):
        prev_observed_data = data[data['Age'] == age_low][['Year', 'Age', 'HIV_female', 'HIV_male']]
        prev_sim_data = pd.DataFrame({'Year': prev_analyzer.timevec, 'Age': age_low, 'sim_HIV_female': prev_results[f'HIV_prev_female_{index}'],
                                 'sim_HIV_male': prev_results[f'HIV_prev_male_{index}']})
        merged = pd.merge(prev_observed_data, prev_sim_data, on=['Year', 'Age'], how='inner')
        merged['error'] = abs(merged['sim_HIV_female'] - merged['HIV_female']) + abs(merged['sim_HIV_male'] - merged['HIV_male'])

        fit += merged['error'].sum()

    return fit


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    # Define the calibration parameters. These are parsed in build_sim() as: {hiv/nw}_{parameter_name}
    # where hiv is for STIsim HIV parameters and nw is for StructuredSexual network parameters.
    calib_pars = dict(
        hiv_beta_m2f = dict(low=0.01, high=0.10, guess=0.05), # HIV transmission parameter
        hiv_beta_m2c = dict(low=0.001, high=0.1, guess=0.025), # Network females in risk group 1 concurrent partners
    )

    calib = run_calib(calib_pars=calib_pars, total_trials=10, keep_db=False)

    sc.toc(T)
    print('Done.')