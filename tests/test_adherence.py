"""
Minimal working adherence test using real STI-Sim HIV
and the same structure as mighti_main.py, but simplified.

This test:
  - creates a valid Starsim + STI-Sim simulation
  - initializes HIV properly (with CD4)
  - applies AdherenceEngine, ARTAdherenceDisruptor,
    InterventionAdherenceDisruptor
  - runs for 1 year
  - checks that adherence modifies ART retention
"""

import numpy as np
import pandas as pd
import starsim as ss
import stisim as sti
import mighti as mi

region = "eswatini"
n_agents=1000
csv_prevalence        = f"mighti/data/{region}_prevalence.csv"
csv_path_params      = f"mighti/data/{region}_parameters.csv"
healthcondition = "MajorDepressiveDisorder"
diseases = ['HIV', healthcondition]

prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases=diseases,
    prevalence_data=prevalence_data_df,
    inityear=2000,
)

def get_prevalence_function(disease):
    def prevalence_func(sim, uids, size=None):
        return mi.age_sex_dependent_prevalence(
            disease=disease,
            prevalence_data=prevalence_data,
            age_bins=age_bins,
            sim=sim,
            uids=uids,
        )
    return prevalence_func

def make_init_prev_func(disease):
    prev_func_local = get_prevalence_function(disease)
    return lambda sim, uids, size=None: prev_func_local(sim, uids, size)



# ---------------------------------------------------------------------
# TEST 1: AdherenceEngine computes adherence < 1 for affected agents
# ---------------------------------------------------------------------
def test_adherence_engine_basic():
    extra_states = [
        ss.FloatArr("adherence", default=1.0),
        ss.BoolArr("neighbourhood_situation"),
    ]

    hiv = sti.HIV(
        beta_m2f=0.2,
        beta_m2c=0.01,
        init_prev=0.20,
    )

    # enable full cascade
    hiv.pars.include_care = True
    hiv.pars.include_aids_deaths = False   # turn off mortality for faster tests
    hiv.pars.art_efficacy = 0.90

    init_prev = ss.bernoulli(p=make_init_prev_func(healthcondition))

    disease_class = getattr(mi, healthcondition, None)

    # Instantiate the disease
    condition_obj = disease_class(
        csv_path=csv_path_params,
        pars={"init_prev": init_prev},
    )

    # Add to the disease list
    disease_objects = [hiv, condition_obj]

    adherence_engine = mi.AdherenceEngine(casm_rel=mi.CASM_REL_FACTORS)

    sim = ss.Sim(
        n_agents=n_agents,
        start=2000,
        stop=2001,
        people=ss.People(n_agents=n_agents, extra_states=extra_states),
        diseases=disease_objects,
        modules=[adherence_engine],
        interventions=[],
        connectors=[],
        label="test_adherence_engine",
    )

    sim.run()

    adher = sim.people.states["adherence"]
    print(adher.mean())
    assert adher.mean() < 1.0, "Adherence should be reduced for CASM-affected agents"

    print("TEST 1 PASSED: AdherenceEngine basic behavior is correct.")

# ---------------------------------------------------------------------
# TEST 2: ARTAdherenceDisruptor increases ART dropout when adherence < 1
# ---------------------------------------------------------------------
def test_art_adherence_disruptor():
    # Same extra_states pattern as TEST 1
    extra_states = [
        ss.FloatArr("adherence", default=1.0),
        ss.BoolArr("neighbourhood_situation"),
    ]

    # --------------------------
    # HIV with full cascade
    # --------------------------
    hiv = sti.HIV(
        beta_m2f=0.2,
        beta_m2c=0.01,
        init_prev=0.20,
    )
    hiv.pars.include_care = True
    hiv.pars.include_aids_deaths = False
    hiv.pars.art_efficacy = 0.90
    # Everyone who is HIV-positive starts ART at t=0
    hiv.pars.init_art = ss.bernoulli(p=1.0)

    # --------------------------
    # MajorDepressiveDisorder (CASM condition)
    # Use the same pattern as TEST 1
    # --------------------------
    disease_class = getattr(mi, healthcondition, None)
    assert disease_class is not None, f"{healthcondition} not found in mighti"

    init_prev_mdd = ss.bernoulli(p=make_init_prev_func(healthcondition))

    mdd_obj = disease_class(
        csv_path=csv_path_params,
        pars={"init_prev": init_prev_mdd},
    )

    disease_objects = [hiv, mdd_obj]

    # --------------------------
    # People + modules
    # --------------------------
    ppl = ss.People(n_agents=n_agents, extra_states=extra_states)

    adherence_engine = mi.AdherenceEngine(casm_rel=mi.CASM_REL_FACTORS)
    art_disruptor    = mi.ARTAdherenceDisruptor(base_dropout=0.40)

    sim = ss.Sim(
        n_agents=n_agents,
        start=2000,
        stop=2001,
        people=ppl,
        diseases=disease_objects,
        modules=[adherence_engine, art_disruptor],
        interventions=[],
        connectors=[],
        label="test_art_disruptor",
    )

    sim.run()

    st = sim.people.states
    adher = st["adherence"]
    on_art = st["hiv.on_art"]
    hiv_infected= sim.diseases.hiv.infected  # HIV-positive

    # Check that adherence actually dropped on average
    print("Mean adherence after run (TEST 2):", adher.mean())
    assert adher.mean() < 1.0, "Adherence should be reduced by CASM conditions."

    # Dropouts = HIV-positive who are no longer on ART
    dropped = (~on_art & hiv_infected).sum()
    print("Number of HIV-positive agents who dropped ART:", dropped)

    assert dropped > 0, "Some HIV-positive agents should drop ART when adherence < 1."

    print("TEST 2 PASSED: ARTAdherenceDisruptor caused ART dropout under low adherence.")

# ---------------------------------------------------------------------
# TEST 3: InterventionAdherenceDisruptor scales REAL ART intervention
# ---------------------------------------------------------------------
def test_intervention_adherence_disruptor():
    print("\nRunning TEST 3 (InterventionAdherenceDisruptor with real ART)…")

    extra_states = [
        ss.FloatArr("adherence", default=1.0),
        ss.BoolArr("neighbourhood_situation"),
    ]
    ppl = ss.People(n_agents=n_agents, extra_states=extra_states)

    hiv = sti.HIV(
        beta_m2f=0.2,
        beta_m2c=0.01,
        init_prev=0.10,
    )
    hiv.pars.include_care = True
    hiv.pars.include_aids_deaths = False
    hiv.pars.art_efficacy = 0.95   # BASELINE

    disease_class = getattr(mi, healthcondition)
    init_prev_mdd = ss.bernoulli(p=make_init_prev_func(healthcondition))
    mdd = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev_mdd})

    art = mi.ARTwithCASM(
        coverage_data=pd.DataFrame({"p_art": [1.0]}, index=[2000])
    )
    art.casm_sensitivity = "pharma"

    adherence_engine = mi.AdherenceEngine(casm_rel=mi.CASM_REL_FACTORS)
    intv_disruptor  = mi.InterventionAdherenceDisruptor()

    sim = ss.Sim(
        n_agents=n_agents,
        start=2000,
        stop=2001,
        people=ppl,
        demographics=[ss.Pregnancy(), ss.Deaths()],
        diseases=[hiv, mdd],
        modules=[adherence_engine, intv_disruptor],
        interventions=[art],
        connectors=[],
        label="test_intv_disruptor",
        copy_inputs=False,
    )

    sim.run()

    # Correct assertion for Starsim 3
    assert hiv.pars.art_efficacy < 0.95, (
        f"Expected HIV ART efficacy < 0.95 after adherence scaling, got {hiv.pars.art_efficacy}"
    )

    print(f"TEST 3 PASSED: ART efficacy scaled to {hiv.pars.art_efficacy:.3f}")


# ---------------------------------------------------------------------
# Run all tests if file executed directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_adherence_engine_basic()
    test_art_adherence_disruptor()
    test_intervention_adherence_disruptor()

    print("\nAll adherence tests passed successfully.")
