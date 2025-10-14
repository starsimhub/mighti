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

To run: `python mighti_cea.py`
"""


import logging
import mighti as mi
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
n_agents = 1_000 
inityear = 2007
endyear = 2020
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

# SDoH 
csv_path_sdoh = f'mighti/data/sdoh.csv'


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

# healthconditions = [condition for condition in df.condition if condition != "HIV"]
# healthconditions = [condition for condition in df.condition if condition not in ["HIV", "HPV", "Flu", "ViralHepatitis"]]
healthconditions = ["Type2Diabetes"]
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
prevalence_analyzer = mi.PrevalenceAnalyzer_HIV(prevalence_data=prevalence_data, diseases=diseases)
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

# Analyzers
microcosting_analyzer_base = mi.MicrocostingAnalyzer(
    unit_costs={'art': 50}, 
    disability_weights={'hiv': 0.2},
    discount_rate_costs=0.03,
    discount_rate_outcomes=0.03,
    name='microcostinganalyzer'
)
microcosting_analyzer_intv = mi.MicrocostingAnalyzer(
    unit_costs={'art': 50}, 
    disability_weights={'hiv': 0.2}, 
    discount_rate=0.03,
    discount_rate_costs=0.03,
    discount_rate_outcomes=0.03,
    name='microcostinganalyzer' )

intervention_analyzer = mi.InterventionAnalyzer(interventions=['art'], name='intervention_analyzer')

death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
    conditions=['hiv'],
    condition_attr_map={'hiv': 'infected'},
    ex_life_expectancy=80
)

analyzers_base = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, 
                  intervention_analyzer, death_cause_analyzer, microcosting_analyzer_base]

analyzers_intv = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, 
                  intervention_analyzer, death_cause_analyzer, microcosting_analyzer_intv]


# ---------------------------------------------------------------------
# Demographics and Networks
# ---------------------------------------------------------------------
death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = ss.Deaths(death_rates) 
death.death_rate_data *= 0.4 # 0.4 for only T2D
fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

# SDoH states
extra_sdoh_states = [
    ss.BoolArr('neighbourhood_situation'),
    ss.BoolArr('social_context'),
    ss.BoolArr('education_situation'),
    ss.BoolArr('economic_situation'),
    ss.BoolArr('healthcare_system'),
]

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age), extra_states=extra_sdoh_states)

maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]


# ---------------------------------------------------------------------
# SDoH
# ---------------------------------------------------------------------
sdoh_modules = [
    mi.NeighbourhoodSituation(csv_path=csv_path_sdoh,condition_name='NeighbourhoodSituation'),
    mi.SocialContext(csv_path=csv_path_sdoh,condition_name='SocialContext'),
    mi.EducationSituation(csv_path=csv_path_sdoh,condition_name='EducationSituation'),
    mi.EconomicSituation(csv_path=csv_path_sdoh,condition_name='EconomicSituation'),
    mi.HealthCareSystem(csv_path=csv_path_sdoh,condition_name='HealthCareSystem'),
]

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
connectors = [ncd_hiv_connector]

ncd_interactions = mi.read_interactions(csv_path_interactions) 
connectors.extend(mi.create_connectors(ncd_interactions))


# -------------------------
# Adherence
# -------------------------

# adherence_connectors = [
#     mi.create_adherence_connector('T2D_Tx'),
#     mi.create_adherence_connector('ART'),
# ]
# interactions.extend(adherence_connectors)


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
unified_product = ss.Tx(df=intervention_df, label='UnifiedTx')


hiv_test = sti.HIVTest(test_prob_data=test_prob_data, years=test_years)
art = sti.ART(coverage_data=art_coverage_data)
vmmc = sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}})
prep = sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]})

interventions1 = [hiv_test, art, vmmc, prep]


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
    sim_base = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=analyzers_base,
        diseases=disease_objects,
        # connectors=connectors + sdoh_modules,
        label='Baseline'
    )

    sim_intv = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=analyzers_intv,
        diseases=disease_objects,
        # connectors=connectors + sdoh_modules,
        interventions=[hiv_test, art],
        label='With ART'
    )

    msim = ss.MultiSim([sim_base, sim_intv])
    msim.run()
    
    analyzer_base = sim_base.analyzers.microcostinganalyzer
    analyzer_intv = sim_intv.analyzers.microcostinganalyzer
    
    # # Compute ICER
    icer = analyzer_intv.compute_icer(analyzer_base)
    
    # Print results
    df_art = sim_intv.analyzers.intervention_analyzer.to_df()
    n_art = df_art[df_art['received_art'] == True]['uid'].nunique()
    print(f" {n_art:,} agents received ART during the intervention.")
    
    
    print("\nIncremental Cost-Effectiveness Results:")
    print(f"  ΔCost  = ${icer['delta_cost']:,.2f}")
    print(f"  ΔDALY  = {icer['delta_daly']:,.2f}")
    print(f"  ICER   = ${icer['icer']:,.2f} per DALY averted")
    
    
