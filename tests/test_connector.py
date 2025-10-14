"""
HIV–AUD Connector Test using MIGHTI and StarSim 3.x MultiSim

Compares two simulations:
Baseline: HIV + AlcoholUseDisorder, no connector.
Connector: HIV + AlcoholUseDisorder + NCDHIVConnector.

The connector should increase or maintain AlcoholUseDisorder prevalence.
"""

import os
import pandas as pd
import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti
import mighti as mi

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
sc.options(interactive=False)
thisdir = os.path.dirname(__file__)
region = "eswatini"
inityear, endyear = 2007, 2020
n_agents = 3000

param_path = os.path.join(thisdir, "test_data", f"{region}_parameters.csv")
prev_path  = os.path.join(thisdir, "test_data", f"{region}_prevalence.csv")

# ---------------------------------------------------------------------
# Load parameter & prevalence data
# ---------------------------------------------------------------------
params_df = pd.read_csv(param_path)
params_df.columns = params_df.columns.str.strip()

prev_df = pd.read_csv(prev_path)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases=["HIV", "AlcoholUseDisorder"],
    prevalence_data=prev_df,
    inityear=inityear,
)

def get_prev_func(disease):
    """Return StarSim 3.x-compatible prevalence callable."""
    def func(sim, uids):
        return mi.age_sex_dependent_prevalence(
            disease=disease,
            prevalence_data=prevalence_data,
            age_bins=age_bins,
            sim=sim,
            size=uids,
        )
    return func

# ---------------------------------------------------------------------
# Build disease modules
# ---------------------------------------------------------------------
hiv = sti.HIV()
hiv_prev = get_prev_func("HIV")
hiv.pars.init_prev = ss.bernoulli(p=lambda sim, uids: hiv_prev(sim, uids))
hiv.pars.beta = {"structuredsexual": [0.03, 0.03]}
hiv.pars.include_aids_deaths = True
hiv.pars.include_care = False
hiv.pars.art_efficacy = 0.0
hiv.pars.p_hiv_death = ss.bernoulli(p=0.00015)

aud = mi.AlcoholUseDisorder(csv_path=param_path, init_prev=ss.bernoulli(p=0.10))

# ---------------------------------------------------------------------
# Networks, demographics, connectors
# ---------------------------------------------------------------------
networks = sti.StructuredSexual()
# death = ss.Deaths({"death_rate": pd.DataFrame({"year": [2007], "death_rate": [0.01]}), "rate_units": 1})
# pregnancy = ss.Pregnancy(pars={"fertility_rate": pd.DataFrame({"year": [2007], "fertility_rate": [0.02]})})
ppl = ss.People(n_agents=n_agents)  # minimal, avoids external CSVs

connector = mi.NCDHIVConnector({"alcoholusedisorder": 2.47})

# ---------------------------------------------------------------------
# Build two sims (baseline vs connector)
# ---------------------------------------------------------------------
sim_base = ss.Sim(
    n_agents=n_agents,
    start=inityear,
    stop=endyear,
    people=ppl,
    networks=networks,
    diseases=[hiv, aud],
    label="Baseline (no connector)",
)

sim_conn = ss.Sim(
    n_agents=n_agents,
    start=inityear,
    stop=endyear,
    people=ppl,
    networks=networks,
    diseases=[hiv, aud],
    connectors=[connector],
    label="With HIV–AUD connector",
)

# ---------------------------------------------------------------------
# MultiSim wrapper and run
# ---------------------------------------------------------------------
def test_hiv_aud_connector_msim():
    msim = ss.MultiSim([sim_base, sim_conn])
    msim.run()

    # Extract mean AUD prevalence
    prev_base = msim.sims[0].results.alcoholusedisorder.prevalence.mean()
    prev_conn = msim.sims[1].results.alcoholusedisorder.prevalence.mean()

    msg = f"AUD prevalence should be ≥ baseline with connector: {prev_base:.3f} → {prev_conn:.3f}"
    assert prev_conn >= prev_base - 0.01, msg
    print(f"[✓] {msg}")
    return msim


# ---------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_hiv_aud_connector_msim()
