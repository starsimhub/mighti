"""
MIGHTI Life Expectancy Over Time Simulation Script for Eswatini

This script uses the MIGHTI framework to simulate and compare the evolution of
life expectancy over time across multiple health intervention scenarios.

Here, we loop over years to generate a time series of life expectancy estimates
under different conditions.

Key features:
- Computes life expectancy (ex) annually by sex and scenario.
- Supports intervention comparison: No intervention, HIV only, T2D only, and combined HIV + T2D.
- Integrates demographic modules (age structure, mortality, fertility).
- Includes HIV and Type 2 Diabetes disease modules and interactions.
- Applies time-varying interventions (ART, HIV testing, VMMC, PrEP, and T2D treatment).
- Uses MIGHTI’s life table utilities to compute sex-specific and combined e₀ values.
"""

import logging
import mighti as mi
import numpy as np
import pandas as pd
import prepare_data_for_year
import os
import starsim as ss
import stisim as sti
from mighti.diseases.type2diabetes import T2D_ReduceMortalityTx
from data_prep.us_data.region_data_builder import ensure_region_data
from mighti.plot_functions import plot_life_expectancy_timeseries
from mighti.rng import seed_everything


# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 

SEED = 12345
seed_everything(SEED)


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
n_agents = 1_000 
inityear = 2007
endyear = 2030
region = 'eswatini'

# ---- knobs (to make outputs interpretable) ----
# If you want to see the historical Eswatini HIV-era dip (~1990s), you will need:
# - an earlier inityear (e.g. 1980) and
# - HIV mortality enabled (include_aids_deaths=True) and
# - a plausible HIV epidemic trajectory (from data or generated endogenously).
DEATH_RATE_SCALER = 1.0           # avoid arbitrary scaling unless you know why
INCLUDE_AIDS_DEATHS = True        # must be True to see HIV-driven mortality impacts
HIV_INIT_PREV_FALLBACK = 0.01     # used only if prevalence CSV has no HIV columns


# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
# Parameters
csv_path_params = f'mighti/data/{region}_parameters.csv'

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus.csv"

# Disease prevalence data
csv_prevalence = f'mighti/data/{region}_prevalence.csv'

# Fertility data 
csv_path_fertility = f'mighti/data/{region}_asfr.csv'

# Death data
csv_path_death = f'mighti/data/{region}_mortality_rates.csv'

# Age distribution data
csv_path_age = f'mighti/data/{region}_age_distribution_{inityear}.csv'

# Intervention 
csv_path_intervention = f'mighti/data/{region}_intervention.csv'

# Ensure required demographic files are prepared
ensure_region_data(region=region, start_year=inityear, end_year=endyear, overwrite=False)
prepare_data_for_year.prepare_data_for_year(region,inityear)
prepare_data_for_year.prepare_data(region)

# Data paths for post process
mx_path = f'mighti/data/{region}_mx.csv'
ex_path = f'mighti/data/{region}_ex.csv'


# ---------------------------------------------------------------------
# Load Parameters and Disease Configuration
# ---------------------------------------------------------------------
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

healthconditions = ['Type2Diabetes']
diseases = ["HIV"] + healthconditions

# ---------------------------------------------------------------------
# Prevalence Data and Analyzers
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, prevalence_data=prevalence_data_df, inityear=inityear
)

# Detect whether HIV prevalence exists in the prevalence CSV (many MIGHTI prevalence tables omit HIV)
_has_hiv_prev_cols = any("hiv" in c.lower() for c in prevalence_data_df.columns)

def make_init_prev_func(disease):
    """Return a Starsim-compatible callable p(sim, uids, size=None) for ss.bernoulli()."""
    def prevalence_func(sim, uids, size=None):
        return mi.age_sex_dependent_prevalence(
            disease=disease,
            prevalence_data=prevalence_data,
            age_bins=age_bins,
            sim=sim,
            uids=uids,
        )
    return prevalence_func


# ---------------------------------------------------------------------
# Demographics and Networks
# ---------------------------------------------------------------------
def make_sim(year):
    # Initialize the PrevalenceAnalyzer
    prevalence_analyzer = mi.analyzers.PrevalenceAnalyzer_HIV(prevalence_data=prevalence_data, diseases=diseases)
    survivorship_analyzer = mi.analyzers.SurvivorshipAnalyzer(label="survivorship_analyzer")
    deaths_analyzer = mi.analyzers.DeathsByAgeSexAnalyzer(label="deaths_by_age_sex_analyzer")
    
    # ConditionAtDeathAnalyzer in this repo does not take `condition_attr_map`
    # (and it handles HIV separately), so keep this minimal.
    death_cause_analyzer = mi.analyzers.ConditionAtDeathAnalyzer(conditions=['hiv', 'type2diabetes'])
    death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
    death = ss.Deaths(death_rates) 
    if DEATH_RATE_SCALER != 1.0:
        death.death_rate_data *= float(DEATH_RATE_SCALER)
    fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    
    ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))
    
    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]

    
    hiv_init_prev = ss.bernoulli(p=make_init_prev_func("HIV")) if _has_hiv_prev_cols else ss.bernoulli(p=HIV_INIT_PREV_FALLBACK)
    hiv_disease = sti.HIV(
        init_prev=hiv_init_prev,
        include_aids_deaths=bool(INCLUDE_AIDS_DEATHS),
        beta={
            'structuredsexual': [0.011023883426646121, 0.011023883426646121],
            'maternal': [0.044227226248848076, 0.044227226248848076],
        },
    )
        # Best pars: {'hiv_beta_m2f': 0.011023883426646121, 'hiv_beta_m2c': 0.044227226248848076} seed: 12345
    
    disease_objects = []
    for dis in healthconditions:
        cls = getattr(mi.diseases, dis, None)
        if cls is not None:
            disease_objects.append(
                cls(csv_path=csv_path_params, pars={"init_prev": ss.bernoulli(p=make_init_prev_func(dis))})
            )
    disease_objects.append(hiv_disease)
    
    
    ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
    ncd_hiv_connector = mi.interactions.NCDHIVConnector(ncd_hiv_rel_sus)
    interactions = [ncd_hiv_connector]
    
    ncd_interactions = mi.interactions.read_interactions(csv_path_interactions) 
    connectors = mi.interactions.create_connectors(ncd_interactions)
    
    interactions.extend(connectors)
        
            
    # ART coverage among PLHIV (from 95-95-95 cascade estimates and Lancet data)
    art_coverage_data = pd.DataFrame({
        'p_art': [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]
    }, index=[2003, 2010, 2013, 2014, 2016, 2022])

    # HIV testing probabilities over time (estimated testing uptake)
    test_prob_data = [0.10, 0.25, 0.60, 0.70, 0.80, 0.95]
    test_years = [2003, 2005, 2007, 2010, 2014, 2016]

    intervention_df = pd.read_csv(csv_path_intervention)
    unified_product = ss.Tx(df=intervention_df, label='UnifiedTx')

    # Some intervention CSVs may not include rows for every disease used in this script.
    # Create minimal fallback Tx definitions so `treat_num`-style interventions don't crash.
    def _make_tx_for(disease_key: str, label: str):
        df_tx = intervention_df.copy()
        if "disease" in df_tx.columns:
            df_tx["disease"] = df_tx["disease"].astype(str).str.lower()
            df_tx = df_tx[df_tx["disease"] == disease_key.lower()]

        if df_tx.empty:
            df_tx = pd.DataFrame(
                {
                    "disease": [disease_key.lower()],
                    "state": ["affected"],
                    "post_state": ["on_treatment"],
                    "efficacy": [1.0],
                }
            )
        return ss.Tx(df=df_tx, label=label)


    hiv_test = sti.HIVTest(test_prob_data=test_prob_data, years=test_years)
    art = sti.ART(coverage_data=art_coverage_data)
    vmmc = sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}})
    prep = sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]})

    t2d_tx_product = _make_tx_for("type2diabetes", label="T2D_Tx")
    t2d_tx = mi.diseases.T2D_ReduceMortalityTx(product=t2d_tx_product, prob=1.0, rel_death_reduction=0.54,
                                              eligibility=lambda sim: sim.diseases.type2diabetes.affected.uids,
                                              label='T2D_ReduceMortalityTx')

    depression_tx_product = _make_tx_for("majordepressivedisorder", label="Depression_Tx")
    depression_tx = mi.diseases.DepressionCare(product=depression_tx_product, prob=0.1, label='depression_tx')

    hospital_discharge = mi.interventions.ImproveHospitalDischarge(disease_name='depression', multiplier=10.0,
                                                                   start_day=0, end_day=10, label='FastDischarge')

    give_housing = mi.interventions.GiveHousingToDepressed(coverage=1, start_day=0)

    # Define interventions using these data
    intervention_hiv = [hiv_test, art, vmmc, prep]
    intervention_t2d = [t2d_tx]
    intervention_both = [hiv_test, art, vmmc, prep, t2d_tx]



    sim_with_hiv = ss.Sim(
        rand_seed=SEED,
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=year,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer],
        diseases=disease_objects,
        connectors=interactions,
        interventions = intervention_hiv,
        copy_inputs=False,
        label='HIV intervention'
    )
    
    
    # ### To run 2 simulation simultaneously #####
    sim_without = ss.Sim(
        rand_seed=SEED,
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=year,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
        diseases=disease_objects,
        connectors=interactions,
        # interventions = interventions,
        copy_inputs=False,
        label='No_intervention'
    )
    
 
    msim = ss.MultiSim(sims=[sim_with_hiv, sim_without])

    return msim



# ---------------------------------------------------------------------
# Utility: Get Modules
# ---------------------------------------------------------------------
def get_deaths_module(sim):
    """Fetch the DeathsByAgeSexAnalyzer from sim.analyzers (not sim.modules)."""
    analyzers = getattr(sim, "analyzers", None)
    if analyzers is None:
        raise ValueError("Simulation has no analyzers attached.")

    # dict / odict-like
    if hasattr(analyzers, "values"):
        for a in analyzers.values():
            if isinstance(a, mi.analyzers.DeathsByAgeSexAnalyzer):
                return a
        # fallback by common keys
        if hasattr(analyzers, "get"):
            for key in ("deaths_by_age_sex_analyzer", "deaths_analyzer", "deathsbyagesexanalyzer"):
                a = analyzers.get(key, None)
                if a is not None:
                    return a
    else:
        # list-like
        for a in analyzers:
            if isinstance(a, mi.analyzers.DeathsByAgeSexAnalyzer):
                return a

    raise ValueError("DeathsByAgeSexAnalyzer not found in sim.analyzers.")

def get_pregnancy_module(sim):
    for module in sim.modules:
        if isinstance(module, ss.Pregnancy):
            return module
    raise ValueError("Pregnancy module not found in the simulation.")


years = list(range(inityear+1, endyear))

life_expectancy_by_year = []

# Run MultiSim
for year in years:
    msim = make_sim(year)

    msim.run(parallel=False)

    for sim in msim.sims:
        label = sim.label
        deaths_module = get_deaths_module(sim)
        df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=year, max_age=100, radix=n_agents)

        df_male = df_mx[df_mx['sex'] == 'Male']
        df_female = df_mx[df_mx['sex'] == 'Female']
        lt = mi.calculate_life_table_from_mx(sim, df_male, df_female)

        # Male and female life expectancy at birth
        for sex in ['Male', 'Female']:
            e0 = lt[(lt['sex'] == sex) & (lt['Age'] == 0)]['e(x)'].values[0]
            life_expectancy_by_year.append({
                'year': year,
                'scenario': label,
                'sex': sex,
                'e0': e0
            })

        # Both sexes (weighted average)
        lt0 = lt[lt['Age'] == 0].copy()
        total_l0 = lt0['l(x)'].sum()
        lt0['weight'] = lt0['l(x)'] / total_l0
        weighted_e0 = (lt0['e(x)'] * lt0['weight']).sum()
        life_expectancy_by_year.append({
            'year': year,
            'scenario': label,
            'sex': 'Both',
            'e0': weighted_e0
        })

# Convert to DataFrame
le_df = pd.DataFrame(life_expectancy_by_year)

highlights = [1990] if (min(years) <= 1990 <= max(years)) else None
# le_df.to_csv("life_expectancy_timeseries_long.csv", index=False)

# Wide table for easy inspection: year × scenario × sex
pivot_df = le_df.pivot_table(index="year", columns=["scenario", "sex"], values="e0").reset_index()
# pivot_df.to_csv("life_expectancy_timeseries_wide.csv", index=False)

# Plots (pop up figures). Save only if you uncomment the fig.savefig(...) lines below.
for sex in ["Both", "Male", "Female"]:
    fig, ax = plot_life_expectancy_timeseries(
        le_df,
        sex=sex,
        highlight_years=highlights,
        title=f"Eswatini e₀ over time ({sex})",
    )
    # fig.savefig(f"life_expectancy_timeseries_{sex.lower()}.png", dpi=200)