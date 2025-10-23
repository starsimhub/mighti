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


diseases = ["HIV"]

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
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()


analyzers = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer]

# ---------------------------------------------------------------------
# Demographics & networks
# ---------------------------------------------------------------------
death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
death = ss.Deaths(death_rates)

fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

# Main
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Beta sensitivity test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # # Create base population and networks (shared across runs)
    # ppl = ss.People(n_agents=10000)
    # sexual = sti.StructuredSexual(name="structuredsexual")
    # maternal = ss.MaternalNet(name="maternal")
    # networks = [sexual, maternal]

    # death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
    # death = ss.Deaths(death_rates)
    # fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
    # pregnancy = ss.Pregnancy(pars=fertility_rate)
    # demographics = [pregnancy, death]

    # # Beta test combinations
    # beta_tests = [
    #     (0.01, 0.001),
    #     (0.1, 0.01),
    #     (0.5, 0.05),
    # ]

    # print("\n--- HIV Beta Sensitivity Test ---")
    # for beta_m2f, beta_m2c in beta_tests:
    #     hiv = sti.HIV(beta_m2f=beta_m2f, beta_m2c=beta_m2c, init_prev=0.05)
    #     sim = ss.Sim(
    #         n_agents=10000,
    #         start=inityear,
    #         stop=endyear,
    #         people=ppl,
    #         networks=networks,
    #         demographics=demographics,
    #         diseases=[hiv],
    #         label=f"HIV beta_m2f={beta_m2f}, beta_m2c={beta_m2c}",
    #     )
    #     sim.run()
    #     final_prev = sim.diseases.hiv.infected.sum() / sim.people.n_agents
    #     print(f"β_m2f={beta_m2f:<6}, β_m2c={beta_m2c:<6} → Final prevalence = {final_prev:.3f}")

    # print("\n Beta sensitivity test complete.")

    for bmf, bmc in [(0.01, 0.001),(0.1,0.01),(0.5,0.05)]:
        ppl = ss.People(n_agents=10000)  # NEW each loop
        sexual  = sti.StructuredSexual(name="structuredsexual")  # NEW each loop
        maternal = ss.MaternalNet(name="maternal")
        hiv = sti.HIV(beta_m2f=bmf, beta_m2c=bmc, init_prev=0.05)
        sim = ss.Sim(start=2007, stop=2020, people=ppl, networks=[sexual, maternal], diseases=[hiv])
        sim.run()
        print(bmf, bmc, hiv.infected.sum()/sim.people.n_agents)
