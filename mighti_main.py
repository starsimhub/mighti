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
from mighti.plot_functions import plot_mean_prevalence
from mighti.util.plot_style import apply_mighti_style
from mighti.util.rng import seed_everything
from pathlib import Path


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
logger = logging.getLogger("MIGHTI")
logger.setLevel(logging.INFO)

# Reproducibility (example script)
SEED = 12345
seed_everything(SEED)

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
csv_path_sdoh         = f'mighti/data/sdoh.csv'

# Optional post-process targets (commented out below)
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

healthconditions = ['COPD']
# healthconditions = [condition for condition in df.condition if condition != "HIV"]
# healthconditions = [
#     condition
#     for condition in df.condition
#     if condition not in ["HIV", "TB", "HPV", "Flu", "ViralHepatitis"]
# ]
diseases = ["HIV"] + healthconditions

# ---------------------------------------------------------------------
# Read prevalence table and build callable prevalence data
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)
hc_diseases = [d for d in diseases if str(d).lower() != "hiv"]
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases=hc_diseases,
    prevalence_data=prevalence_data_df,
    inityear=inityear,
)

## HIV prevalence comes from a dedicated wide file (Age/Year/HIV_male/HIV_female)
hiv_prev_df = pd.read_csv("mighti/data/eswatini_prevalence_hiv.csv")
hiv_prev_data, hiv_age_bins = mi.initialize_prevalence_data(
    diseases=["HIV"],
    prevalence_data=hiv_prev_df,
    inityear=inityear,
)

def get_prevalence_function(disease):
    def prevalence_func(sim, uids, size=None):
        # Use the dedicated HIV prevalence table so we don't need to mix HIV columns
        # into `eswatini_prevalence.csv`.
        if str(disease).lower() == "hiv":
            return mi.age_sex_dependent_prevalence(
                disease=disease,
                prevalence_data=hiv_prev_data,
                age_bins=hiv_age_bins,
                sim=sim,
                uids=uids,
            )
        return mi.age_sex_dependent_prevalence(
            disease=disease,
            prevalence_data=prevalence_data,
            age_bins=age_bins,
            sim=sim,
            uids=uids,
        )
    return prevalence_func


# ---------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------
prevalence_analyzer = mi.analyzers.PrevalenceAnalyzer_HIV(
    prevalence_data=prevalence_data,  # ok for non-HIV diseases; HIV plotting uses sim results
    diseases=diseases,
)
# survivorship_analyzer = mi.SurvivorshipAnalyzer()
# deaths_analyzer = mi.DeathsByAgeSexAnalyzer()
# death_cause_analyzer = mi.ConditionAtDeathAnalyzer(conditions=healthconditions)
# analyzers = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer]


# ---------------------------------------------------------------------
# Demographics & networks
# ---------------------------------------------------------------------
death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
death = ss.Deaths(death_rates)

fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

# People with adherence state
extra_states = [
    ss.FloatArr("adherence", default=1.0),
    ss.BoolArr("neighbourhood_situation"),
    # ss.BoolArr("social_context"),
    # ss.BoolArr("education_situation"),
    # ss.BoolArr("economic_situation"),
    # ss.BoolArr("healthcare_system"),
]

# Build People from the empirical age–sex distribution (no file outputs)
ppl = mi.people_extend.make_people_with_age_sex(
    csv_path=csv_path_age,
    init_year=inityear,
    n_agents=n_agents,
    out_dir=None,
    extra_states=extra_states,
)

maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]

# ---------------------------------------------------------------------
# Adherence modules
# ---------------------------------------------------------------------
adherence_engine = mi.AdherenceEngine(
    casm_rel=mi.CASM_REL_FACTORS,
    sdoh_rel={},  # can add SDoH keys → multipliers later
)
art_disruptor = mi.ARTAdherenceDisruptor(base_dropout=0.10)
intervention_disruptor = mi.InterventionAdherenceDisruptor()

sdoh_modules = mi.sdoh.NeighbourhoodSituation(csv_path=csv_path_sdoh)
# ---------------------------------------------------------------------
# Diseases
# ---------------------------------------------------------------------
disease_objects = []

# --- HIV ---
# hiv = sti.HIV(
#     beta_m2f=0.002824808975498053,
#     beta_m2c=0.0015347394768786338,
#     init_prev=0.15,
# )

hiv = sti.HIV(
    beta_m2f=0.03362975603278965,
    beta_m2c=0.008587993382382253,
    init_prev=0.15,
)

# Assign prevalence as age–sex dependent
prev_func = get_prevalence_function("HIV")
hiv.pars.init_prev = ss.bernoulli(
    p=lambda sim, uids, size=None: prev_func(sim, uids, size)
)

# AIDS mortality and care
hiv.pars.include_aids_deaths = True
hiv.pars.p_hiv_death = ss.bernoulli(p=0.00015)
hiv.pars.include_care = True
hiv.pars.art_efficacy = 0.9

disease_objects.append(hiv)

# Other diseases from CSV
def make_init_prev_func(disease):
    prev_func_local = get_prevalence_function(disease)
    return lambda sim, uids, size=None: prev_func_local(sim, uids, size)

for disease in healthconditions:
    disease_class = getattr(mi.diseases, disease, None)
    if disease_class is None:
        continue
    init_prev = ss.bernoulli(p=make_init_prev_func(disease))
    disease_obj = disease_class(
        csv_path=csv_path_params,
        pars={"init_prev": init_prev},
    )
    disease_objects.append(disease_obj)


# ---------------------------------------------------------------------
# Connectors (HIV ↔ NCD, plus other NCD interactions)
# ---------------------------------------------------------------------
ncd_hiv_rel_sus = df.set_index("condition")["rel_sus"].to_dict()
ncd_hiv_connector = mi.interactions.NCDHIVConnector(ncd_hiv_rel_sus)
connectors = [ncd_hiv_connector]

ncd_interactions = mi.interactions.read_interactions(csv_path_interactions)
connectors.extend(mi.interactions.create_connectors(ncd_interactions))


# ---------------------------------------------------------------------
# Interventions (ART + HIV testing)
# ---------------------------------------------------------------------
art_coverage_data = pd.DataFrame(
    {"p_art": [0.10, 0.34, 0.50, 0.65, 0.74, 0.85, 0.95]},
    index=[2003, 2010, 2013, 2014, 2016, 2022, 2050],
)

hiv_test = sti.HIVTest(
    test_prob_data=[0.10, 0.25, 0.60, 0.70, 0.80, 0.95, 0.95],
    years=[2003, 2005, 2007, 2010, 2014, 2016, 2050],
)

art = mi.ARTwithCASM(coverage_data=art_coverage_data)
art.casm_sensitivity = "pharma"  # interpreted by CASMAdherenceConnector
art.rel_effect = 1.0             # baseline; adherence scales this

# Example of other intervention products (not used in interventions_hiv yet)
intervention_df = pd.read_csv(csv_path_intervention)
unified_product = ss.Tx(df=intervention_df, label="UnifiedTx")

interventions_hiv = [hiv_test, art]


# ---------------------------------------------------------------------
# Optional helper for mortality analyzers (kept for future use)
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
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_mighti_style()

    # dt=1: one timestep per year. Disease p_acquire is per-timestep (no scaling), so dt=1
    # keeps prevalence from being inflated by multiple draws per year.
    sim = ss.Sim(
        rand_seed=SEED,
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        dt=1,
        people=ppl,
        networks=networks,
        demographics=[pregnancy, death],
        diseases=disease_objects,
        connectors=connectors,
        modules=[adherence_engine, art_disruptor, intervention_disruptor, sdoh_modules],
        interventions=interventions_hiv,  # HIV test + ARTwithCASM
        analyzers=[prevalence_analyzer],  # must be a list
        copy_inputs=False,
        label="With Interventions",
    )

    # Run simulation
    sim.run()

    # Prevalence plot: use a disease that is actually in the sim (first non-HIV)
    plot_disease = next((d for d in diseases if str(d).lower() != "hiv"), diseases[0] if diseases else "Type2Diabetes")
    prevalence_check_df = pd.read_csv(
        f"mighti/data/{region}_postprocess_check_prevalence.csv"
    )
    plot_mean_prevalence(
        sim,
        prevalence_analyzer,
        plot_disease,
        prevalence_check_df,
        inityear,
        endyear,
    )

    import pandas as pd
    from mighti.plot_functions import plot_hiv_prevalence_vs_observed

    obs = pd.read_csv("mighti/data/eswatini_prevalence_hiv.csv")

    # after you run sim and have access to the prevalence analyzer object
    plot_hiv_prevalence_vs_observed(
        sim,
        prevalence_analyzer,
        obs,
        age_starts=[15, 20, 25, 30, 35, 40, 45],  # pick bins you want
        start_year=1990,
        end_year=2023,
    )
    from mighti.plot_functions import plot_mean_prevalence_plhiv
    plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'COPD')  