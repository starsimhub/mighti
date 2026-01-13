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
import numpy as np

# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 



# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
logger = logging.getLogger("MIGHTI")
logger.setLevel(logging.INFO)

n_agents = 100_000
inityear = 2007
endyear = 2020
region = "eswatini"

# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
csv_path_params       = f"mighti/data/{region}_parameters.csv"
csv_path_interactions = "mighti/data/rel_sus.csv"
csv_prevalence        = f"mighti/data/{region}_prevalence.csv"
csv_path_fertility    = f"mighti/data/{region}_asfr.csv"
csv_path_death        = f"mighti/data/{region}_mortality_rates.csv"
csv_path_age          = f"mighti/data/{region}_age_distribution_{inityear}.csv"
csv_path_intervention = f"mighti/data/{region}_intervention.csv"
csv_path_sdoh = f'mighti/data/sdoh.csv'

# Post-process targets
mx_path = f"mighti/data/{region}_mx.csv"
ex_path = f"mighti/data/{region}_ex.csv"

# Ensure required demographic files exist
prepare_data_for_year.prepare_data_for_year(region, inityear)
prepare_data_for_year.prepare_data(region)

# ---------------------------------------------------------------------
# Load parameters & define which diseases to include
# ---------------------------------------------------------------------
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

# Keep it minimal for debugging: HIV + one HC
healthconditions = ["Type2Diabetes"]
diseases = ["HIV"] + healthconditions

#---------------------------------------------------------------------
# Read prevalence table and build callable prevalence data
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases=diseases, prevalence_data=prevalence_data_df, inityear=inityear
)

def get_prevalence_function(disease):
    def prevalence_func(sim, uids, size=None):
        return mi.age_sex_dependent_prevalence(
            disease=disease, prevalence_data=prevalence_data,
            age_bins=age_bins, sim=sim, size=size,
        )
    return prevalence_func


# ---------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------
prevalence_analyzer = mi.PrevalenceAnalyzer_HIV(prevalence_data=prevalence_data, diseases=diseases)
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
    conditions=healthconditions)

analyzers = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer]


# Analyzers
microcosting_analyzer_base = mi.MicrocostingAnalyzer(
    unit_costs={'art': 50}, 
    disability_weights={'hiv': 0.2, 'type2diabetes': 0.1},
    discount_rate_costs=0.03,
    discount_rate_outcomes=0.03,
    name='microcostinganalyzer'
)
microcosting_analyzer_intv = mi.MicrocostingAnalyzer(
    unit_costs={'art': 50}, 
    disability_weights={'hiv': 0.2, 'type2diabetes': 0.1},
    discount_rate=0.03,
    discount_rate_costs=0.03,
    discount_rate_outcomes=0.03,
    name='microcostinganalyzer' )

intervention_analyzer = mi.InterventionAnalyzer(interventions=['art'], name='intervention_analyzer')

analyzers_base = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, 
                  intervention_analyzer, death_cause_analyzer, microcosting_analyzer_base]

analyzers_intv = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, 
                  intervention_analyzer, death_cause_analyzer, microcosting_analyzer_intv]

# ---------------------------------------------------------------------
# Demographics & networks
# ---------------------------------------------------------------------
maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]

death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
death = ss.Deaths(death_rates)

fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))



# ---------------------------------------------------------------------
# Diseases 
# ---------------------------------------------------------------------
disease_objects = []

# --- HIV ---
hiv = sti.HIV()

# Assign prevalence
prev_func = get_prevalence_function('HIV')
hiv.pars.init_prev = ss.bernoulli(
    p=lambda sim, uids, size=None: prev_func(sim, uids, size)
)

# Transmission parameters
# Best pars: {'hiv_beta_m2f': 0.09553835265049065, 'hiv_beta_m2c': 0.003895160642773216}
# Best pars: {'hiv_beta_m2f': 0.041126225026336546, 'hiv_beta_m2c': 0.02313161100759324}
hiv.pars.beta = {
    'structuredsexual': [0.029594299274445842, 0.029594299274445842],
    'maternal': [0.0011249414706988527, 0.0011249414706988527],
}

disease_objects.append(hiv)


def make_init_prev_func(disease):
    prev_func = get_prevalence_function(disease)
    return lambda sim, uids, size=None: prev_func(sim, uids, size)

# Other diseases
for disease in healthconditions:
    disease_class = getattr(mi, disease, None)
    if disease_class:
        init_prev = ss.bernoulli(p=make_init_prev_func(disease))
        disease_obj = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev})
        disease_objects.append(disease_obj)


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
        connectors=connectors,
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
        connectors=connectors,
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

    cost_base = analyzer_base.results.total_cost
    cost_art = analyzer_intv.results.total_cost
    daly_base = analyzer_base.results.total_daly
    daly_art = analyzer_intv.results.total_daly

    daly_averted = daly_base - daly_art
    cost_increment = cost_art - cost_base

    icer = cost_increment / daly_averted if daly_averted > 0 else np.inf

    print("\n ICER Calculation:")
    print(f"  Cost (baseline): ${cost_base:,.2f}")
    print(f"  Cost (ART):      ${cost_art:,.2f}")
    print(f"  DALY (baseline): {daly_base:,.2f}")
    print(f"  DALY (ART):      {daly_art:,.2f}")
    print(f"  DALYs averted:   {daly_averted:,.2f}")
    print(f"  Incremental Cost: ${cost_increment:,.2f}")
    print(f"  ICER: ${icer:,.2f} per DALY averted")

    d = sim_intv.diseases.type2diabetes
    dur = d.duration
    print("NaNs:", np.isnan(dur).sum(), "mean duration:", np.mean(dur))
    import inspect

    diab = sim_intv.diseases.get('type2diabetes', None)
    diab2 = sim_base.diseases.get('type2diabetes', None)

    print('--- DIABETES DEBUG ---')
    print('Class:', diab.__class__)
    print('MRO:', inspect.getmro(diab.__class__))
    print('Has duration attr:', hasattr(diab, 'duration'))

    if hasattr(diab, 'duration'):
        print('duration type:', type(diab.duration))
        print('first few values:', diab.duration[:10])

    print('--- DIABETES DEBUG (Base) ---')
    print('Class:', diab2.__class__)
    print('MRO:', inspect.getmro(diab2.__class__))
    print('Has duration attr:', hasattr(diab2, 'duration'))

    if hasattr(diab2, 'duration'):
        print('duration type:', type(diab2.duration))
        print('first few values:', diab2.duration[:10])

    summary_base = mi.summarize_microcosting_results(analyzer_base)
    summary_intv = mi.summarize_microcosting_results(analyzer_intv)

    print("\nSummary: Baseline")
    for k, v in summary_base.items():
        print(f"{k}: {v:,.2f}")

    print("\nSummary: With ART")
    for k, v in summary_intv.items():
        print(f"{k}: {v:,.2f}")


    # Example usage for current run
    results = [{
        'label': 'ART vs Baseline',
        'delta_daly': 1012498.05 - 632424.29,
        'delta_cost': 7558117.26 - 0
    }]
    mi.plot_cost_effectiveness_plane(results)