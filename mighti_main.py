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
# from mighti.diseases.type2diabetes import ReduceMortalityTx


# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
n_agents = 10_000 
inityear = 2007  
endyear = 2010
region = 'eswatini'


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

def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)


# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
    conditions=['hiv', 'type2diabetes'],
    condition_attr_map={
        'hiv': 'infected',
        'type2diabetes': 'affected'  
    }
)

# ---------------------------------------------------------------------
# Demographics and Networks
# ---------------------------------------------------------------------
death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = ss.Deaths(death_rates) 
fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]


# ---------------------------------------------------------------------
# Diseases
# ---------------------------------------------------------------------
hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')),
                      init_prev_data=None,   
                      p_hiv_death=None, 
                      include_aids_deaths=False, 
                      beta={'structuredsexual': [0.011023883426646121, 0.011023883426646121], 
                            'maternal': [0.044227226248848076, 0.044227226248848076]})
    # Best pars: {'hiv_beta_m2f': 0.011023883426646121, 'hiv_beta_m2c': 0.044227226248848076} seed: 12345

disease_objects = []

for disease in healthconditions:
    disease_class = getattr(mi, disease, None)
    if disease_class:
        init_prev = ss.bernoulli(get_prevalence_function(disease))
        disease_obj = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev})
        disease_objects.append(disease_obj)
        
disease_objects.append(hiv_disease)


# ---------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------
ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
interactions = [ncd_hiv_connector]

ncd_interactions = mi.read_interactions(csv_path_interactions) 
connectors = mi.create_connectors(ncd_interactions)

interactions.extend(connectors)


# ---------------------------------------------------------------------
# Interventions 
# ---------------------------------------------------------------------
# ART coverage among PLHIV (from 95-95-95 cascade estimates and Lancet data)
art_coverage_data = pd.DataFrame({
    'p_art': [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]
}, index=[2003, 2010, 2013, 2014, 2016, 2022])

# HIV testing probabilities over time (estimated testing uptake)
test_prob_data = [0.10, 0.25, 0.60, 0.70, 0.80, 0.95]
test_years = [2003, 2005, 2007, 2010, 2014, 2016]

intervention_df = pd.read_csv(csv_path_intervention)
intervention = ss.Tx(df=intervention_df)

# Define interventions using these data
interventions = [
    sti.HIVTest(test_prob_data=test_prob_data, years=test_years),
    sti.ART(coverage_data=art_coverage_data),
    sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
    sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]}),
]

interventions2 = [
    sti.HIVTest(test_prob_data=test_prob_data, years=test_years),
    sti.ART(coverage_data=art_coverage_data),
    sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
    sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]}),
    mi.T2D_ReduceMortalityTx(product=intervention,prob=1.0,rel_death_reduction=0.54,
                             eligibility=lambda sim: sim.diseases.type2diabetes.affected.uids)
]

interventions3 = [
    mi.T2D_ReduceMortalityTx(product=intervention,prob=1.0,rel_death_reduction=0.54,
                             eligibility=lambda sim: sim.diseases.type2diabetes.affected.uids)
]

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


# ---------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------
if __name__ == '__main__':
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
        diseases=disease_objects,
        connectors=interactions,
        # interventions = interventions3,
        copy_inputs=False,
        label='With Interventions'
    )
    # Run the simulation
    sim.run()
    
    
    
    # # Mortality rates and life table
    # target_year = endyear - 1
    
    # obs_mx = prepare_data_for_year.extract_indicator_for_plot(mx_path, target_year, value_column_name='mx')
    # obs_ex = prepare_data_for_year.extract_indicator_for_plot(ex_path, target_year, value_column_name='ex')
    
    # # Get the modules
    # deaths_module = get_deaths_module(sim)
    # pregnancy_module = get_pregnancy_module(sim)
    
    # df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=target_year, max_age=100, radix=n_agents)

    # df_mx_male = df_mx[df_mx['sex'] == 'Male']
    # df_mx_female = df_mx[df_mx['sex'] == 'Female']
    
    
    # life_table = mi.calculate_life_table_from_mx(sim, df_mx_male, df_mx_female, max_age=100)
    
    # mi.plot_mx_comparison(df_mx, obs_mx, year=target_year, age_interval=5)
    
    # # Plot life expectancy comparison
    # mi.plot_life_expectancy(life_table, obs_ex, year = target_year, max_age=100, figsize=(14, 10), title=None)
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, inityear, endyear)
    
    # df = death_cause_analyzer.to_df()   
    # df['HIV only'] = df['died_hiv'] & ~df['died_type2diabetes']
    # df['T2D only'] = df['died_type2diabetes'] & ~df['died_hiv']
    # df['Both'] = df['died_hiv'] & df['died_type2diabetes']
    # df['Neither'] = ~df['died_hiv'] & ~df['died_type2diabetes']
    # counts = df[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
    # print(counts)
    # df.groupby('sex')[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
    
    # # df[['had_hiv', 'died_of_hiv', 'had_type2diabetes', 'died_of_type2diabetes']].sum()
    
       
    # #### To run 2 simulation simultaneously #####
    # sim_without = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     # interventions = interventions,
    #     copy_inputs=False,
    #     label='No Intervention'
    # )
    
    # sim_with = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     interventions = interventions,
    #     copy_inputs=False,
    #     label='With HIV Intervention'
    # )
    
    # sim_with_both = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     interventions = interventions2,
    #     copy_inputs=False,
    #     label='With HIV and T2D ntervention'
    # )
 
    # msim = ss.MultiSim(sims=[sim_without, sim_with, sim_with_both])
    # msim.run()
    
    # # Target year for evaluation
    # target_year = endyear - 1
    
    # # Load observed mortality and life expectancy
    # obs_mx = prepare_data_for_year.extract_indicator_for_plot(mx_path, target_year, value_column_name='mx')
    # obs_ex = prepare_data_for_year.extract_indicator_for_plot(ex_path, target_year, value_column_name='ex')
    
    # # Helper to extract mortality rates and life table from one sim
    # def process_life_table(sim):
    #     deaths_module = get_deaths_module(sim)
    #     df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=target_year, max_age=100, radix=n_agents)
    #     df_mx_male = df_mx[df_mx['sex'] == 'Male']
    #     df_mx_female = df_mx[df_mx['sex'] == 'Female']
    #     life_table = mi.calculate_life_table_from_mx(sim, df_mx_male, df_mx_female, max_age=100)
    #     return df_mx, life_table
    
    # # Process both sims in MultiSim
    # df_mx_without, lt_without = process_life_table(msim.sims[0])
    # df_mx_with, lt_with = process_life_table(msim.sims[1])
    # df_mx_with_both, lt_with_both = process_life_table(msim.sims[2])
    
    # # Plot mx comparison (can pick one to compare to observed)
    # mi.plot_mx_comparison(df_mx_with, obs_mx, year=target_year, age_interval=5)
    
    # # Plot life expectancy: Sim with vs. without vs. Observed
    # mi.plot_life_expectancy_four(
    #     sim_no_intervention=lt_without,
    #     sim_hiv_only=lt_with,
    #     sim_both_interventions=lt_with_both,
    #     observed_data=obs_ex,
    #     year=target_year
    # )    
