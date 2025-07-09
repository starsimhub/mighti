"""
MIGHTI Simulation Script for IAS
"""


import logging
import mighti as mi
import numpy as np
import pandas as pd
import prepare_data_for_year
import starsim as ss
import stisim as sti
from mighti.diseases.type2diabetes import ReduceMortalityTx


# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
n_agents = 100_000 
inityear = 2007  
endyear = 2050
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
csv_path_death = f'mighti/data/{region}_mortality_rates.csv'

# Age distribution data
csv_path_age = f'mighti/data/{region}_age_distribution_{inityear}.csv'

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


# ---------------------------------------------------------------------
# Interventions 
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


def make_hiv_interventions():
    return [
        sti.HIVTest(test_prob_data=test_prob_data, years=test_years),
        sti.ART(coverage_data=art_coverage_data),
        sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
        sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]}),
    ]

def make_t2d_interventions():
    tx_df = pd.read_csv("mighti/data/t2d_tx.csv")
    t2d_tx = ss.Tx(df=tx_df)
    t2d_treatment = ReduceMortalityTx(
        label='T2D Mortality Reduction',
        product=t2d_tx,
        prob=1.0,
        rel_death_reduction=0.5,
        eligibility=lambda sim: sim.diseases.type2diabetes.affected.uids
    )
    return [t2d_treatment]

def make_combined_interventions():
    return make_hiv_interventions() + make_t2d_interventions()


def make_sim(interventions, label, seed):
    ss.set_seed(seed)  # Ensures Starsim's RNG is properly seeded


    # Fresh people object for each run
    people = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))
    fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    # Reload prevalence data to generate fresh distribution functions
    prevalence_data_df = pd.read_csv(csv_prevalence)
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases, prevalence_data=prevalence_data_df, inityear=inityear
    )

    def get_prev_dist(disease):
        return ss.bernoulli(
            p=lambda mod, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)
        )

    # Fresh analyzers
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

    # Networks
    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]

    # Diseases
    hiv_disease = sti.HIV(
        init_prev=get_prev_dist('HIV'),
        init_prev_data=None,
        p_hiv_death=None,
        include_aids_deaths=False,
        beta={
            'structuredsexual': [0.011023883426646121, 0.011023883426646121],
            'maternal': [0.044227226248848076, 0.044227226248848076]
        }
    )

    t2d_disease = mi.Type2Diabetes(
        csv_path=csv_path_params,
        pars={"init_prev": get_prev_dist("Type2Diabetes")}
    )

    disease_objects = [hiv_disease, t2d_disease]

    # Interactions
    ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
    ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
    interactions = [ncd_hiv_connector]

    ncd_interactions = mi.read_interactions(csv_path_interactions)
    connectors = mi.create_connectors(ncd_interactions)
    interactions.extend(connectors)

    # Assemble and return sim
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=people,
        demographics=[pregnancy],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
        diseases=disease_objects,
        connectors=interactions,
        interventions=interventions,
        copy_inputs=False,
        label=label
    )
    
    print("\n[Post-Sim Debug]")
    print("Example ti_dead values for HIV and T2D:")
    
    # print("HIV ti_dead (non-nan):", np.sum(~np.isnan(sim.people.hiv.ti_dead)))
    # print("T2D ti_dead (non-nan):", np.sum(~np.isnan(sim.people.type2diabetes.ti_dead)))
    print(f"Sim {label} with seed={seed}")
    print(f"Initial HIV+ agents: {np.sum(hiv_disease.infected)}")
    print(f"Initial T2D+ agents: {np.sum(t2d_disease.affected)}")
    return sim, death_cause_analyzer


def run_sim_and_get_deaths(intervention_fn, label, seed):
    if intervention_fn is None:
        interventions = None
    else:
        interventions = intervention_fn()

    sim, analyzer = make_sim(interventions, label, seed)
    sim.run()

    df = analyzer.to_df()
    df['HIV only'] = df['died_hiv'] & ~df['died_type2diabetes']
    df['T2D only'] = df['died_type2diabetes'] & ~df['died_hiv']
    df['Both'] = df['died_hiv'] & df['died_type2diabetes']
    df['Neither'] = ~df['died_hiv'] & ~df['died_type2diabetes']

    counts = df[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
    # Now safe to access
    print(f"Running {label} with seed={seed}")
    print(f"Initial HIV+ agents: {np.sum(sim.diseases.hiv.infected)}")
    print(f"Initial T2D+ agents: {np.sum(sim.diseases.type2diabetes.affected)}")
    return counts

def run_replicates(n_runs, intervention_fn, label):
    results = []
    for i in range(n_runs):
        counts = run_sim_and_get_deaths(intervention_fn, label, seed=i)
        results.append(counts)

    df = pd.DataFrame(results)
    print("[CI Debug] Raw T2D only death counts across replicates:")
    print(df['T2D only'].values)
    return df.describe(percentiles=[0.025, 0.5, 0.975]).T[['2.5%', '50%', '97.5%']]

run_num = 10
ci_no   = run_replicates(run_num, intervention_fn=None, label='No Intervention')
ci_hiv  = run_replicates(run_num, intervention_fn=make_hiv_interventions, label='HIV Intervention')
ci_t2d  = run_replicates(run_num, intervention_fn=make_t2d_interventions, label='T2D Intervention')
ci_both = run_replicates(run_num, intervention_fn=make_combined_interventions, label='Both Interventions')

print("No Intervention:\n", ci_no)
print("HIV Intervention:\n", ci_hiv)
print("T2D Intervention:\n", ci_t2d)
print("Both Interventions:\n", ci_both)


# Initialize the PrevalenceAnalyzer


# # ---------------------------------------------------------------------
# # Demographics and Networks
# # ---------------------------------------------------------------------
# death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
# death = ss.Deaths(death_rates) 
# fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
# pregnancy = ss.Pregnancy(pars=fertility_rate)

# ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

# maternal = ss.MaternalNet()
# structuredsexual = sti.StructuredSexual()
# networks = [maternal, structuredsexual]


# # ---------------------------------------------------------------------
# # Diseases
# # ---------------------------------------------------------------------
# hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prev_fn('HIV')),
#                       init_prev_data=None,   
#                       p_hiv_death=None, 
#                       include_aids_deaths=False, 
#                       beta={'structuredsexual': [0.011023883426646121, 0.011023883426646121], 
#                             'maternal': [0.044227226248848076, 0.044227226248848076]})
#     # Best pars: {'hiv_beta_m2f': 0.011023883426646121, 'hiv_beta_m2c': 0.044227226248848076} seed: 12345

# t2d_disease = mi.Type2Diabetes(csv_path=csv_path_params,pars={"init_prev": ss.bernoulli(p=get_prev_fn("Type2Diabetes"))})

# # Add to the disease object list
# disease_objects = [hiv_disease, t2d_disease]


# # ---------------------------------------------------------------------
# # Interactions
# # ---------------------------------------------------------------------
# ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
# ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
# interactions = [ncd_hiv_connector]

# ncd_interactions = mi.read_interactions(csv_path_interactions) 
# connectors = mi.create_connectors(ncd_interactions)

# interactions.extend(connectors)



# # ---------------------------------------------------------------------
# # Utility: Get Modules
# # ---------------------------------------------------------------------
# def get_deaths_module(sim):
#     for module in sim.modules:
#         if isinstance(module, mi.DeathsByAgeSexAnalyzer):
#             return module
#     raise ValueError("Deaths module not found in the simulation. Make sure you've added the DeathsByAgeSexAnalyzer to your simulation configuration")

# def get_pregnancy_module(sim):
#     for module in sim.modules:
#         if isinstance(module, ss.Pregnancy):
#             return module
#     raise ValueError("Pregnancy module not found in the simulation.")


# def run_sim_and_get_deaths(interventions, label, seed=None):
#     if seed is not None:
#         np.random.seed(seed)

#     # Create fresh analyzers for each run
#     deaths_analyzer = mi.DeathsByAgeSexAnalyzer()
#     survivorship_analyzer = mi.SurvivorshipAnalyzer()
#     prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=['Type2Diabetes', 'HIV'])
#     death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
#         conditions=['hiv', 'type2diabetes'],
#         condition_attr_map={'hiv': 'infected', 'type2diabetes': 'affected'}
#     )

#     sim = ss.Sim(
#         n_agents=n_agents,
#         networks=networks,
#         start=inityear,
#         stop=endyear,
#         people=ppl,
#         demographics=[pregnancy],
#         analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
#         diseases=disease_objects,
#         connectors=interactions,
#         interventions=interventions,
#         copy_inputs=False,
#         label=label
#     )
#     sim.run()


#     # Get analyzer from sim
#     analyzer = next((a for a in sim.analyzers if isinstance(a, mi.ConditionAtDeathAnalyzer)), None)
#     if analyzer is None:
#         raise RuntimeError("ConditionAtDeathAnalyzer not found in sim")

#     df = analyzer.to_df()
#     return df

# def run_replicates(n_runs, interventions, label):
#     results = []
#     for i in range(n_runs):
#         df = run_sim_and_get_deaths(interventions, label, seed=i)
#         df['HIV only'] = df['died_hiv'] & ~df['died_type2diabetes']
#         df['T2D only'] = df['died_type2diabetes'] & ~df['died_hiv']
#         df['Both'] = df['died_hiv'] & df['died_type2diabetes']
#         df['Neither'] = ~df['died_hiv'] & ~df['died_type2diabetes']
#         counts = df[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
#         results.append(counts)

#     results_df = pd.DataFrame(results)
#     return results_df.describe(percentiles=[0.025, 0.5, 0.975]).T[['2.5%', '50%', '97.5%']]

# ci_no = run_replicates(50, None, 'No Intervention')
# ci_hiv = run_replicates(50, interventions, 'HIV Intervention')
# ci_t2d = run_replicates(50, interventions3, 'T2D Intervention')
# ci_both = run_replicates(50, interventions2, 'Both Interventions')

# print("No Intervention:\n", ci_no)
# print("HIV Intervention:\n", ci_hiv)
# print("T2D Intervention:\n", ci_t2d)
# print("Both Interventions:\n", ci_both)


# # if __name__ == '__main__':
# #     sim = ss.Sim(
# #         n_agents=n_agents,
# #         networks=networks,
# #         start=inityear,
# #         stop=endyear,
# #         people=ppl,
# #         demographics=[pregnancy],
# #         analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
# #         diseases=disease_objects,
# #         connectors=interactions,
# #         interventions = interventions2,
# #         copy_inputs=False,
# #         label='With Interventions'
# #     )
# #     # Run the simulation
# #     sim.run()

# #     df = death_cause_analyzer.to_df()   
# #     df['HIV only'] = df['died_hiv'] & ~df['died_type2diabetes']
# #     df['T2D only'] = df['died_type2diabetes'] & ~df['died_hiv']
# #     df['Both'] = df['died_hiv'] & df['died_type2diabetes']
# #     df['Neither'] = ~df['died_hiv'] & ~df['died_type2diabetes']
# #     counts = df[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
# #     print(counts)




# # def make_sim(label, interventions):
# #     return ss.Sim(
# #         n_agents=n_agents,
# #         networks=networks,
# #         start=inityear,
# #         stop=endyear,
# #         people=ppl,
# #         demographics=[pregnancy],
# #         analyzers=[
# #             mi.DeathsByAgeSexAnalyzer(),
# #             mi.SurvivorshipAnalyzer(),
# #             mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=['Type2Diabetes', 'HIV']),
# #             mi.ConditionAtDeathAnalyzer(
# #                 name='conditionatdeathanalyzer',
# #                 conditions=['hiv', 'type2diabetes'],
# #                 condition_attr_map={'hiv': 'infected', 'type2diabetes': 'affected'}
# #             )
# #         ],
# #         diseases=disease_objects,
# #         connectors=interactions,
# #         interventions=interventions,
# #         copy_inputs=False,
# #         label=label
# #     )

# # # Define sims
# # sim_no = make_sim('No Intervention', interventions=None)
# # sim_hiv = make_sim('HIV Intervention', interventions=interventions)
# # sim_t2d = make_sim('T2D Intervention', interventions=interventions3)
# # sim_both = make_sim('Both Interventions', interventions=interventions2)

# # # Run MultiSim
# # msim = ss.MultiSim([sim_no, sim_hiv, sim_t2d, sim_both])
# # msim.run()

# # # Utility function to get an analyzer by name
# # def get_analyzer_by_name(sim, name):
# #     for analyzer in sim.analyzers:
# #         if getattr(analyzer, 'name', '') == name:
# #             return analyzer
# #     raise ValueError(f"Analyzer '{name}' not found in sim '{sim.label}'")

# # # Process results
# # for sim in msim.sims:
# #     # Get the correct analyzer
# #     death_analyzer = get_analyzer_by_name(sim, 'conditionatdeathanalyzer')
# #     df = death_analyzer.to_df()

# #     # Compute death categories
# #     df['HIV only'] = df['died_hiv'] & ~df['died_type2diabetes']
# #     df['T2D only'] = df['died_type2diabetes'] & ~df['died_hiv']
# #     df['Both'] = df['died_hiv'] & df['died_type2diabetes']
# #     df['Neither'] = ~df['died_hiv'] & ~df['died_type2diabetes']
# #     counts = df[['HIV only', 'T2D only', 'Both', 'Neither']].sum()

# #     print(f'\n{sim.label}:\n{counts}')
    