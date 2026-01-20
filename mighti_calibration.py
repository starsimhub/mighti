"""
Calibrate betas for HIV
"""


import optuna
import mighti as mi
import pandas as pd
import sciris as sc
import starsim as ss
import stisim as sti
import logging

logger = logging.getLogger(__name__)


def make_sim():
  
    hiv = sti.HIV(beta_m2f=0.05, beta_m2c=0.025, init_prev=0.15)
    fertility_rate = {'fertility_rate': pd.read_csv('mighti/data/eswatini_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    death_rates = {'death_rate': pd.read_csv('mighti/data/eswatini_mortality_rates.csv'), 'rate_units': 1}
    death = ss.Deaths(death_rates)  # Assuming death_rate is a yearly rate

    sexual = sti.StructuredSexual()
    maternal = ss.MaternalNet()

    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=pd.read_csv('mighti/data/eswatini_prevalence.csv'), diseases=['HIV'])

    sim = ss.Sim(
        dt=1,
        n_agents=10_000,
        total_pop=9_980_999,
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
        reseed=False,
        sampler=optuna.samplers.TPESampler(seed=12345) 
    )

    calib.calibrate()
    calib.check_fit()

    # Return the results for further analysisz
    return calib


def eval_fn(sim, data=None, sim_result_list=None, weights=None, df_res_list=None):
    """
    Custom evaluation function for HIV calibration
    """
    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]

    def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    if data is None:
        raise ValueError("eval_fn requires observed HIV prevalence data via eval_kw={'data': ...}.")

    hiv_female_col = _find_col(data, ["HIV_female", "hiv_female"])
    hiv_male_col = _find_col(data, ["HIV_male", "hiv_male"])
    if hiv_female_col is None or hiv_male_col is None:
        hiv_like = [c for c in data.columns if "hiv" in c.lower()]
        raise ValueError(
            "Observed prevalence data is missing required columns for HIV by sex. "
            f"Expected columns like 'HIV_female' and 'HIV_male'. "
            f"Found HIV-like columns: {hiv_like}. "
            "Please provide an HIV prevalence CSV with Age/Year and sex-stratified HIV prevalence."
        )

    # Normalize observed data (if in %)
    if pd.to_numeric(data[hiv_female_col], errors="coerce").max() > 1:
        data[[hiv_female_col, hiv_male_col]] = data[[hiv_female_col, hiv_male_col]] / 100.0

    fit = 0
    prev_analyzer = sim.analyzers.get('prevalence_analyzer') if hasattr(sim.analyzers, "get") else getattr(sim.analyzers, "prevalence_analyzer", None)
    prev_results = sim.results.get('prevalence_analyzer') if hasattr(sim, "results") else None
    if prev_analyzer is None or prev_results is None:
        raise ValueError("PrevalenceAnalyzer results not found on sim; ensure analyzers include mi.PrevalenceAnalyzer with label 'prevalence_analyzer'.")

    for index, (age_low, age_high) in enumerate(prev_analyzer.age_bins):
        prev_observed_data = data[data['Age'] == age_low][['Year', 'Age', hiv_female_col, hiv_male_col]].copy()
        prev_observed_data['Year'] = prev_observed_data['Year'].astype(int)

        # Normalize analyzer time vector to int years
        sim_years = [t.year if hasattr(t, 'year') else int(t) for t in prev_analyzer.timevec]

        prev_sim_data = pd.DataFrame({
            'Year': sim_years,
            'Age': age_low,
            'sim_HIV_female': prev_results[f'hiv_prev_female_{index}'],
            'sim_HIV_male':   prev_results[f'hiv_prev_male_{index}'],
        })

        merged = pd.merge(prev_observed_data, prev_sim_data, on=['Year', 'Age'], how='inner')
        merged['error'] = abs(merged['sim_HIV_female'] - merged[hiv_female_col]) + \
                        abs(merged['sim_HIV_male'] - merged[hiv_male_col])
        fit += merged['error'].sum()

    n_obs = len(data['Age'].unique()) * 2
    return fit / n_obs


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    # Define the calibration parameters. These are parsed in build_sim() as: {hiv/nw}_{parameter_name}
    # where hiv is for STIsim HIV parameters and nw is for StructuredSexual network parameters.
    calib_pars = dict(
        hiv_beta_m2f = dict(low=0.001, high=0.10, guess=0.03), # HIV transmission parameter
        hiv_beta_m2c = dict(low=0.0001, high=0.1, guess=0.001), # Network females in risk group 1 concurrent partners
    )

    calib = run_calib(calib_pars=calib_pars, total_trials=100, keep_db=False)

    sc.toc(T)
    print('Done.')
    