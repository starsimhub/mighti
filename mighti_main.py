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
n_agents = 100_000 
inityear = 2007  
endyear = 2024
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
# death_cause_analyzer = mi.ConditionAtDeathAnalyzer(conditions=["HIV", "Type2Diabetes"])

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
hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prev_fn('HIV')),
                      init_prev_data=None,   
                      p_hiv_death=0.00001, 
                      include_aids_deaths=False, 
                      beta={'structuredsexual': [0.011023883426646121, 0.011023883426646121], 
                            'maternal': [0.044227226248848076, 0.044227226248848076]})
    # Best pars: {'hiv_beta_m2f': 0.011023883426646121, 'hiv_beta_m2c': 0.044227226248848076} seed: 12345

disease_objects = []
for dis in healthconditions:
    cls = getattr(mi, dis, None)
    if cls is not None:
        disease_objects.append(cls(csv_path=csv_path_params, pars={"init_prev": ss.bernoulli(get_prev_fn(dis))}))
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
# Interventions (with updated empirical data)
# ---------------------------------------------------------------------
# ART coverage among PLHIV (from 95-95-95 cascade estimates and Lancet data)
art_coverage_data = pd.DataFrame({
    'p_art': [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]
    # 'p_art': [1,1,1,1,1,1]
}, index=[2003, 2010, 2013, 2014, 2016, 2022])

# HIV testing probabilities over time (estimated testing uptake)
test_prob_data = [0.10, 0.25, 0.60, 0.70, 0.80, 0.95]
# test_prob_data = [1,1,1,1,1,1]
test_years = [2003, 2005, 2007, 2010, 2014, 2016]

# Define interventions using these data
interventions = [
    sti.HIVTest(test_prob_data=test_prob_data, years=test_years),
    sti.ART(coverage_data=art_coverage_data),
    sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
    sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]})
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
        interventions = interventions,
        copy_inputs=False,
        label='Connector'
    )
 
    # Run the simulation
    sim.run()

    
    df = death_cause_analyzer.to_df()   
    df['HIV only'] = df['had_hiv'] & ~df['had_type2diabetes']
    df['T2D only'] = df['had_type2diabetes'] & ~df['had_hiv']
    df['Both'] = df['had_hiv'] & df['had_type2diabetes']
    df['Neither'] = ~df['had_hiv'] & ~df['had_type2diabetes']
    counts = df[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
    print(counts)
    df.groupby('sex')[['HIV only', 'T2D only', 'Both', 'Neither']].sum()

    # # Plot prevalence
    # plot_diseases = ['Type2Diabetes']
    # for disease in plot_diseases:
    #     mi.plot_mean_prevalence(sim, prevalence_analyzer, disease, prevalence_data_df, init_year=inityear, end_year=endyear)
    #     mi.plot_age_group_prevalence(sim, prevalence_analyzer, disease, prevalence_data_df, init_year=inityear, end_year=endyear)
    #     mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease)






    # Mortality rates and life table
    target_year = 2023
    
    obs_mx = prepare_data_for_year.extract_indicator_for_plot(mx_path, target_year, value_column_name='mx')
    obs_ex = prepare_data_for_year.extract_indicator_for_plot(ex_path, target_year, value_column_name='ex')
    
    # Get the modules
    deaths_module = get_deaths_module(sim)
    pregnancy_module = get_pregnancy_module(sim)
    
    df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=target_year, max_age=100, radix=n_agents)

    df_mx_male = df_mx[df_mx['sex'] == 'Male']
    df_mx_female = df_mx[df_mx['sex'] == 'Female']
    
    mi.plot_mx_comparison(df_mx, obs_mx, year=target_year, age_interval=5)

    # Create the life table
    life_table = mi.create_life_table(sim, df_mx_male, df_mx_female, max_age=100, radix=n_agents)
    
    # Plot life expectancy comparison
    mi.plot_life_expectancy(life_table, obs_ex, year = target_year, max_age=100, figsize=(14, 10), title=None)
    
    # Example: Proportion of HIV+ people on ART at end of sim
    infected = sim.people.hiv['infected']
    on_art = sim.people.hiv['on_art']
    
    n_infected = infected.sum()
    n_on_art = (infected & on_art).sum()
    print(f"{n_on_art} / {n_infected} HIV+ people are on ART")
    # print(f"Deaths with HIV: {n_hiv_deaths}")


# # Ensure age and year are integers
# obs_mx['Age'] = obs_mx['Age'].astype(int)
# obs_mx['Time'] = obs_mx['Time'].astype(int)
# df_mx['age'] = df_mx['age'].astype(int)
# df_mx['year'] = df_mx['year'].astype(int)

# # Filter to 2007 only
# obs_mx_2007 = obs_mx[obs_mx['Time'] == 2007]
# df_mx_2007 = df_mx[df_mx['year'] == 2007]

# obs_mx_male = obs_mx_2007[obs_mx_2007['Sex']=='Male']
# obs_mx_female = obs_mx_2007[obs_mx_2007['Sex']=='Female']


# # Pivot to have Age as index, Sex as columns
# obs_pivot = obs_mx_2007.pivot(index='Age', columns='Sex', values='mx')
# sim_pivot = df_mx_2007.pivot(index='age', columns='sex', values='mx')

# # Align both (intersection of available ages and sexes)
# common_index = obs_pivot.index.intersection(sim_pivot.index)
# common_cols = obs_pivot.columns.intersection(sim_pivot.columns)

# obs_mx_matched = obs_pivot.loc[common_index, common_cols]
# sim_mx_matched = sim_pivot.loc[common_index, common_cols]
    
# import matplotlib.pyplot as plt

# ages = common_index

# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# for i, sex in enumerate(['Male', 'Female']):
#     ax = axs[i]
#     ax.plot(ages, obs_mx_matched[sex], label='Observed', linestyle='--', marker='s', color='black')
#     ax.plot(ages, sim_mx_matched[sex], label='Simulated', linestyle='-', color='blue' if sex == 'Male' else 'red')
#     ax.set_title(f'{sex} Mortality Rate (mx) Comparison in 2007')
#     ax.set_ylabel('mx (deaths per person-year)')
#     ax.set_yscale('log')
#     ax.grid(True)
#     ax.legend()

# axs[1].set_xlabel('Age')
# plt.tight_layout()
# plt.show()    


import pandas as pd
import matplotlib.pyplot as plt

df = death_cause_analyzer.to_df()

df['on_art'] = df['uid'].map(lambda uid: sim.people.hiv['on_art'][uid] if uid < sim.people.n_uids else False)
df['group'] = 'Neither'
df.loc[df['had_hiv'] & ~df['on_art'], 'group'] = 'HIV+, not on ART'
df.loc[df['had_hiv'] & df['on_art'], 'group'] = 'HIV+, on ART'

# Group by age
bins = range(0, 101, 10)
df['age_bin'] = pd.cut(df['age'], bins)

group_counts = df.groupby(['age_bin', 'group']).size().unstack().fillna(0)

# Plot
group_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Deaths by HIV & ART Status")
plt.ylabel("Number of deaths")
plt.xlabel("Age group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import numpy as np

# Classify all agents at final timestep
uids = sim.people.uids
age = sim.people.age
sex = sim.people.female.map({True: 'Female', False: 'Male'})
hiv = sim.people.hiv['infected']
on_art = sim.people.hiv['on_art']
alive = sim.people.alive

df = pd.DataFrame({
    'uid': uids,
    'age': age,
    'sex': sex,
    'hiv': hiv,
    'on_art': on_art,
    'alive': alive
})

# Define age bins
df['age_bin'] = pd.cut(df['age'], bins=np.arange(0, 101, 10), right=False)

# Join death data
death_df = death_cause_analyzer.to_df()
death_df['age_bin'] = pd.cut(death_df['age'], bins=np.arange(0, 101, 10), right=False)

# mx = deaths / alive
def compute_mx(by):
    deaths = death_df[by].groupby('age_bin').size()
    pop = df[df.alive][by].groupby('age_bin').size()
    return (deaths / pop).fillna(0)

mx_hiv_pos = compute_mx(df['hiv'])
mx_hiv_neg = compute_mx(~df['hiv'])

plt.figure(figsize=(10, 5))
plt.plot(mx_hiv_pos.index.astype(str), mx_hiv_pos.values, label="HIV+", marker='o')
plt.plot(mx_hiv_neg.index.astype(str), mx_hiv_neg.values, label="HIV-", marker='o')
plt.yscale('log')
plt.xlabel("Age Group")
plt.ylabel("Mortality rate (mx)")
plt.title("Mortality Rate by HIV Status")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# import matplotlib.pyplot as plt

# # Plot l(x)
# plt.plot(life_table[life_table.sex == 'Male']['Age'], life_table[life_table.sex == 'Male']['l(x)'], label='Sim Male')
# plt.plot(life_table[life_table.sex == 'Female']['Age'], life_table[life_table.sex == 'Female']['l(x)'], label='Sim Female')
# plt.yscale('log')
# plt.title("Simulated Survivorship l(x)")
# plt.legend()
# plt.show()

# # Plot mx if you have observed mx
# plt.plot(df_mx_male['age'], df_mx_male['mx'], label='Sim mx Male')
# # If you have `obs_mx_male`, add it
# plt.plot(obs_mx_male['Age'], obs_mx_male['mx'], label='Obs mx Male')
# plt.yscale('log')
# plt.title("Simulated vs Observed Mortality mx")
# plt.legend()
# plt.show()

# # Plot mx if you have observed mx
# plt.plot(df_mx_male['age'], df_mx_male['mx'], label='Sim mx Male')
# # If you have `obs_mx_male`, add it
# plt.plot(obs_mx_male['Age'], obs_mx_male['mx'], label='Obs mx Male')
# plt.yscale('log')
# plt.title("Simulated vs Observed Mortality mx")
# plt.legend()
# plt.show()