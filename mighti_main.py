"""
MIGHTI Simulation Script for a selected region: HIV and Health Conditions Interaction Modeling

This script initializes and runs an agent-based simulation using the MIGHTI framework
(built on StarSim and STI-Sim) to analyze the interplay between HIV and
other health conditions (HCs) in a selected country.

Key updates for Starsim 3.x:
- Callable distributions now use signature (sim, uids) (no `size`).
- Module parameters are set on module instances BEFORE passing to ss.Sim.
- Do not access `sim.diseases[...]` until after sim is constructed.
"""

import logging
import numpy as np
import pandas as pd

import mighti as mi
import prepare_data_for_year
import starsim as ss
import stisim as sti


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
logger = logging.getLogger("MIGHTI")
logger.setLevel(logging.INFO)

n_agents = 10_000
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

# healthconditions = ['Type2Diabetes']
# healthconditions = [condition for condition in df.condition if condition != "HIV"]
healthconditions = [condition for condition in df.condition if condition not in ["HIV", "TB", "HPV", "Flu", "ViralHepatitis"]]
diseases = ["HIV"] + healthconditions

# ---------------------------------------------------------------------
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
# survivorship_analyzer = mi.SurvivorshipAnalyzer()
# deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

# death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
#     conditions=healthconditions)

# analyzers = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer]

# casm_keys = [
#     "majordepressivedisorder.affected",
#     "alcoholuse.affected",
#     "anxiety.affected",
#     "chronicpain.affected",
#     "tobaccouse.affected",
#     "opioiduse.affected",
#     "stimulantuse.affected",
# ]

# analyzers = []
# for c in casm_keys:
#     a = mi.AdherenceAnalyzer(condition_key=c, intervention_key="hiv.on_art")
#     analyzers.append(a)

# ---------------------------------------------------------------------
# Demographics & networks
# ---------------------------------------------------------------------
death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
death = ss.Deaths(death_rates)

fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

# ppl = mi.make_people_with_age_sex(
#     csv_path="mighti/data/eswatini_age_distribution.csv",
#     init_year=inityear,
#     n_agents=n_agents,
# )
ppl = ss.People(n_agents=n_agents)

maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]

# ---------------------------------------------------------------------
# Diseases 
# ---------------------------------------------------------------------
disease_objects = []

# --- HIV ---
hiv = sti.HIV(
    beta_m2f=0.955,
    beta_m2c=0.0039,
    init_prev=0.15,
)

# Assign prevalence
prev_func = get_prevalence_function('HIV')
hiv.pars.init_prev = ss.bernoulli(
    p=lambda sim, uids, size=None: prev_func(sim, uids, size)
)

# Transmission parameters
# Best pars: {'hiv_beta_m2f': 0.09553835265049065, 'hiv_beta_m2c': 0.003895160642773216}
# Best pars: {'hiv_beta_m2f': 0.041126225026336546, 'hiv_beta_m2c': 0.02313161100759324}
# hiv.pars.beta = {
#     'structuredsexual': [0.029594299274445842, 0.029594299274445842],
#     'maternal': [0.0011249414706988527, 0.0011249414706988527],
# }


# --- AIDS mortality ---
hiv.pars.include_aids_deaths = True
hiv.pars.p_hiv_death = ss.bernoulli(p=0.00015)
hiv.pars.include_care = True
hiv.pars.art_efficacy = 0.9

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
# Connectors (HIV ↔ NCD, plus other NCD interactions)
# ---------------------------------------------------------------------
ncd_hiv_rel_sus = df.set_index("condition")["rel_sus"].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
connectors = [ncd_hiv_connector]

ncd_interactions = mi.read_interactions(csv_path_interactions)
connectors.extend(mi.create_connectors(ncd_interactions))

# ---------------------------------------------------------------------
# Interventions (can be toggled on/off)
# ---------------------------------------------------------------------
art_coverage_data = pd.DataFrame(
    {"p_art": [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]},
    index=[2003, 2010, 2013, 2014, 2016, 2022],
)

test_prob_data = [0.10, 0.25, 0.60, 0.70, 0.80, 0.95]
test_years = [2003, 2005, 2007, 2010, 2014, 2016]

intervention_df = pd.read_csv(csv_path_intervention)
unified_product = ss.Tx(df=intervention_df, label="UnifiedTx")

hiv_test = sti.HIVTest(test_prob_data=test_prob_data, years=test_years)
art = mi.ARTwithCASM(coverage_data=art_coverage_data)  
art.casm_sensitivity = "pharma"
vmmc = sti.VMMC(pars={"future_coverage": {"year": 2015, "prop": 0.30}})
prep = sti.Prep(pars={"coverage": [0, 0.05, 0.25], "years": [2007, 2015, 2020]})

# A simple T2D mortality reduction as an example
t2d_tx = mi.T2D_ReduceMortalityTx(
    product=unified_product,
    prob=1.0,
    rel_death_reduction=0.54,
    eligibility=lambda sim: sim.diseases.type2diabetes.affected.uids,
    label="T2D_ReduceMortalityTx",
)

# Choose which intervention set to use in the Sim
interventions = [hiv_test, art, vmmc, prep]  # or add t2d_tx, etc.


# ---------------------------------------------------------------------
# Utility: helpers
# ---------------------------------------------------------------------
def get_deaths_module(sim):
    """
    Return the DeathsByAgeSexAnalyzer or any analyzer with 'death' in its name.
    Works with dict-style sim.analyzers (MIGHTI default).
    """
    if hasattr(sim, "analyzers") and isinstance(sim.analyzers, dict):
        for a in sim.analyzers.values():
            if isinstance(a, mi.DeathsByAgeSexAnalyzer) or "death" in a.label.lower():
                return a
    raise ValueError(
        f"Deaths analyzer not found. Available analyzers: {list(sim.analyzers.keys())}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sim = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        people=ppl,
        networks=networks,
        demographics=[pregnancy, death],
        diseases=disease_objects,      
        # connectors=connectors,
        # interventions=interventions,
        analyzers=prevalence_analyzer,
        copy_inputs=False,
        label="With Interventions",
    )

    # Run
    sim.run()


    # # Mortality & life table
    # target_year = endyear - 1
    # obs_mx = prepare_data_for_year.extract_indicator_for_plot(mx_path, target_year, value_column_name="mx")
    # obs_ex = prepare_data_for_year.extract_indicator_for_plot(ex_path, target_year, value_column_name="ex")

    # deaths_module = get_deaths_module(sim)
    # pregnancy_module = get_pregnancy_module(sim)

    # df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=target_year, max_age=100, radix=n_agents)
    # df_mx_male = df_mx[df_mx["sex"] == "Male"]
    # df_mx_female = df_mx[df_mx["sex"] == "Female"]

    # life_table = mi.calculate_life_table_from_mx(sim, df_mx_male, df_mx_female, max_age=100)
    # mi.plot_mx_comparison(df_mx, obs_mx, year=target_year, age_interval=5)
    # mi.plot_life_expectancy(life_table, obs_ex, year=target_year, max_age=100)

    # # # Optional prevalence plots
    prevalence_check_df = pd.read_csv(f"mighti/data/{region}_postprocess_check_prevalence.csv")
    mi.plot_mean_prevalence(sim, prevalence_analyzer, "HIV", prevalence_check_df, inityear, endyear)
    # mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, "CardiovascularDiseases")

    # mi.plot_adherence_by_condition(sim, analyzers, casm_keys)

    male_prev, female_prev = mi.plot_mean_prevalence(
        sim, prevalence_analyzer, "HIV", prevalence_data_df, init_year=2000, end_year=2020
    )

    for t, pm, pf in zip(sim.timevec, male_prev, female_prev):
        year = t.year if hasattr(t, "year") else int(t)
        print(f"Year {year} | Male: {pm:.2f}% | Female: {pf:.2f}%")
