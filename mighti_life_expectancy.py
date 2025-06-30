"""
MIGHTI Simulation Script for a selected region: HIV and Health Conditions Interaction Modeling

This script initializes and runs an agent-based simulation using the MIGHTI framework
(built on StarSim and STI-Sim) to analyze the interplay between HIV and
other health conditions (HCs) in selected country. 
It loads demographic data, initializes diseases and networks, 
applies interventions, and analyzes prevalence and mortality outcomes for the selected period.

Key components:
- Loads parameters and prevalence data from CSV files.
- Initializes networks: maternal and structured sexual.
- Initializes HIV and HC modules.
- Sets up demographic modules (deaths, pregnancy).
- Applies HIV interventions (e.g., ART, VMMC).
- Computes and plots prevalence, mortality rates, and life expectancy.

To run: `python mighti_main.py`
"""


import logging
import mighti as mi
import numpy as np
import pandas as pd
import prepare_data_for_year
import starsim as ss
import stisim as sti

# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
n_agents = 10_000 
inityear = 2007  
region = 'eswatini'


# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
# Parameters
csv_path_params = f'mighti/data/{region}_parameters_gbd.csv'

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus.csv"

# Disease prevalence data
csv_prevalence = f'mighti/data/{region}_prevalence.csv'

# Fertility data 
csv_path_fertility = f'mighti/data/{region}_asfr.csv'

# Death data
csv_path_death = f'mighti/data/{region}_mortality_rates_{inityear}.csv'

# Age distribution data
csv_path_age = f'mighti/data/{region}_age_distribution_{inityear}.csv'

# Ensure required demographic files are prepared
prepare_data_for_year.prepare_data_for_year(region,inityear)

# Data paths for post process
mx_path = f'mighti/data/{region}_mx.csv'
ex_path = f'mighti/data/{region}_ex.csv'


# ---------------------------------------------------------------------
# Load Parameters and Disease Configuration
# ---------------------------------------------------------------------
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

#healthconditions = [condition for condition in df.condition if condition != "HIV"]
# healthconditions = [condition for condition in df.condition if condition not in ["HIV", "TB", "HPV", "Flu", "ViralHepatitis"]]
# healthconditions = ['Type2Diabetes', 'ChronicKidneyDisease', 'CervicalCancer', 'ProstateCancer', 'RoadInjuries', 'DomesticViolence']
# healthconditions = []
healthconditions = ['Type2Diabetes']
diseases = ["HIV"] + healthconditions

ncd_df = df[df["disease_class"] == "ncd"]
chronic = ncd_df[ncd_df["disease_type"] == "chronic"]["condition"].tolist()
acute = ncd_df[ncd_df["disease_type"] == "acute"]["condition"].tolist()
remitting = ncd_df[ncd_df["disease_type"] == "remitting"]["condition"].tolist()
communicable_diseases = df[df["disease_class"] == "sis"]["condition"].tolist()


# ---------------------------------------------------------------------
# Prevalence Data and Analyzers
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, prevalence_data=prevalence_data_df, inityear=inityear
)
get_prev_fn = lambda d: lambda mod, sim, size: mi.age_sex_dependent_prevalence(d, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()


# ---------------------------------------------------------------------
# Demographics and Networks
# ---------------------------------------------------------------------
def make_sim(year):
    # Initialize the PrevalenceAnalyzer
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
    survivorship_analyzer = mi.SurvivorshipAnalyzer()
    deaths_analyzer = mi.DeathsByAgeSexAnalyzer()
    
    death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
    death = ss.Deaths(death_rates) 
    fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    
    ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))
    
    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]
    
    hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prev_fn('HIV')),
                          init_prev_data=None,   
                          p_hiv_death=None, 
                          include_aids_deaths=True, 
                          beta={'structuredsexual': [0.011023883426646121, 0.011023883426646121], 
                                'maternal': [0.044227226248848076, 0.044227226248848076]})
        # Best pars: {'hiv_beta_m2f': 0.011023883426646121, 'hiv_beta_m2c': 0.044227226248848076} seed: 12345

    disease_objects = []
    for dis in healthconditions:
        cls = getattr(mi, dis, None)
        if cls is not None:
            disease_objects.append(cls(csv_path=csv_path_params, pars={"init_prev": ss.bernoulli(get_prev_fn(dis))}))
    disease_objects.append(hiv_disease)
    
    
    ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
    ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
    interactions = [ncd_hiv_connector]
    
    ncd_interactions = mi.read_interactions(csv_path_interactions) 
    connectors = mi.create_connectors(ncd_interactions)
    
    interactions.extend(connectors)
    
    
    art_coverage_data = pd.DataFrame({
        'p_art': [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]
    }, index=[2003, 2010, 2013, 2014, 2016, 2022])
    
    # HIV testing probabilities over time (estimated testing uptake)
    test_prob_data = [0.10, 0.25, 0.60, 0.70, 0.80, 0.95]
    test_years = [2003, 2005, 2007, 2010, 2014, 2016]
    
    # Define interventions using these data
    interventions = [
        sti.HIVTest(test_prob_data=test_prob_data, years=test_years),
        sti.ART(coverage_data=art_coverage_data),
        sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
        sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]})
    ]


    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=year,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer],
        diseases=disease_objects,
        connectors=interactions,
        interventions = interventions,
        copy_inputs=False,
        label='Connector'
    )
    return sim



# ---------------------------------------------------------------------
# Utility: Get Modules
# ---------------------------------------------------------------------
def get_deaths_module(sim):
    for module in sim.modules:
        if isinstance(module, mi.DeathsByAgeSexAnalyzer):
            return module
    raise ValueError("Deaths module not found in the simulation. Make sure you've added the DeathsByAgeSexAnalyzer to your simulation configuration")

def get_pregnancy_module(sim):
    for module in sim.modules:
        if isinstance(module, ss.Pregnancy):
            return module
    raise ValueError("Pregnancy module not found in the simulation.")


    
years = list(range(2008, 2023))  

life_expectancy_by_year = []
for year in years:
    sim = make_sim(year)
    sim.run()
    
    deaths_module = get_deaths_module(sim)
    df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=year, max_age=100, radix=n_agents)

    df_male = df_mx[df_mx['sex'] == 'Male']
    df_female = df_mx[df_mx['sex'] == 'Female']
    lt = mi.create_life_table(sim, df_male, df_female)

    for sex in ['Male', 'Female']:
        e0 = lt[(lt['sex'] == sex) & (lt['Age'] == 0)]['e(x)'].values[0]
        life_expectancy_by_year.append({'year': year, 'sex': sex, 'e0': e0})
        