"""
Full integration test for MIGHTI (Starsim 3.x compatible)

This test verifies that a full MIGHTI simulation runs end-to-end without error:
- Initializes modules (people, networks, diseases, connectors)
- Runs interventions and analyzers
- Produces valid mortality and prevalence outputs
"""

import inspect
import os
import numpy as np
import pandas as pd
import starsim as ss
import stisim as sti
import mighti as mi
from mighti.interventions.core import _stisim_art_accepts_coverage_kw


def _sti_art(coverage_prop, year=2000):
    if _stisim_art_accepts_coverage_kw():
        return sti.ART(coverage=coverage_prop)
    return sti.ART(coverage_data=pd.DataFrame({"p_art": [coverage_prop]}, index=[year]))


def _sti_vmmc(coverage_prop, year=2000):
    if "coverage" in inspect.signature(sti.VMMC.__init__).parameters:
        return sti.VMMC(coverage=coverage_prop)
    return sti.VMMC(coverage_data=pd.DataFrame({"p_vmmc": [coverage_prop]}, index=[year]))


def _sti_prep():
    if "coverage" in inspect.signature(sti.Prep.__init__).parameters:
        return sti.Prep(coverage={"year": [2007, 2015, 2020], "value": [0, 0.05, 0.25]})
    return sti.Prep(pars={"coverage": [0, 0.05, 0.25], "years": [2007, 2015, 2020]})


def test_full_mighti_simulation():
    """Run a minimal Eswatini simulation to confirm all components integrate."""

    thisdir = os.path.dirname(__file__)
    inityear, endyear = 2007, 2009
    n_agents = 500

    # --- Parameters
    param_path = os.path.join(thisdir, "test_data", "eswatini_parameters.csv")
    df = pd.read_csv(param_path)
    df.columns = df.columns.str.strip()
    healthconditions = [c for c in df.condition if c != "HIV"]
    diseases = ["HIV"] + healthconditions

    # --- Prevalence data
    prev_path = os.path.join(thisdir, "test_data", "eswatini_prevalence.csv")
    prevalence_df = pd.read_csv(prev_path)
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases, prevalence_df, inityear
    )

    def get_prev_fn(disease):
        """
        Return a Starsim-3.x compatible callable for init_prev: func(sim, uids, size=None) -> probs
        """
        def prevalence_func(sim, uids, size=None):
            return mi.age_sex_dependent_prevalence(
                disease=disease,
                prevalence_data=prevalence_data,
                age_bins=age_bins,
                sim=sim,
                uids=uids,
            )
        return prevalence_func

    # --- HIV module
    hiv = sti.HIV(
        init_prev=ss.bernoulli(p=get_prev_fn("HIV")),
        beta={"structuredsexual": [0.01, 0.01], "maternal": [0.01, 0.01]},
        include_aids_deaths=True,
    )

    # --- Other diseases
    disease_objects = [hiv]
    active_diseases = ["HIV"]  # keep analyzers in sync with instantiated modules
    for d in healthconditions:
        cls = getattr(mi, d, None)
        if not cls:
            print(f" Skipping unknown condition: {d}")
            continue
        bins = age_bins.get(d, [])
        if len(bins) < 2:
            print(f" Skipping {d}: no prevalence age bins")
            continue
        init_prev = ss.bernoulli(p=get_prev_fn(d))
        disease_objects.append(cls(csv_path=param_path, pars={"init_prev": init_prev}))
        active_diseases.append(d)

    # --- Connectors / interactions
    rel_sus_path = os.path.join(thisdir, "test_data", "rel_sus.csv")
    rel_sus_df = pd.read_csv(rel_sus_path)
    rel_sus_dict = (
        df.dropna(subset=["rel_sus"])
        .set_index("condition")["rel_sus"]
        .to_dict()
    )
    interactions = [mi.NCDHIVConnector(rel_sus_dict)]

    # --- Demographics & networks
    death_path = os.path.join(thisdir, "test_data", "eswatini_mortality_rates.csv")
    fert_path = os.path.join(thisdir, "test_data", "eswatini_asfr.csv")
    age_path = os.path.join(thisdir, "test_data", "eswatini_age_distribution_2007.csv")

    deaths = ss.Deaths({"death_rate": pd.read_csv(death_path), "rate_units": 1})
    pregnancy = ss.Pregnancy({"fertility_rate": pd.read_csv(fert_path)})
    ppl = ss.People(n_agents, age_data=pd.read_csv(age_path))
    networks = [ss.MaternalNet(), sti.StructuredSexual()]

    # --- Analyzers
    analyzers = [
        mi.DeathsByAgeSexAnalyzer(),
        mi.SurvivorshipAnalyzer(),
        mi.PrevalenceAnalyzer_HIV(prevalence_data=prevalence_data, diseases=active_diseases),
    ]

    interventions = [
        sti.HIVTest(test_prob_data=[0.6, 0.7, 0.95], years=[2000, 2007, 2016]),
        _sti_art(0.95, year=inityear),
        _sti_vmmc(0.30, year=inityear),
        _sti_prep(),
    ]

    # --- Simulation
    sim = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, deaths],
        networks=networks,
        diseases=disease_objects,
        connectors=interactions,
        analyzers=analyzers,
        interventions=interventions,
        copy_inputs=False,
        label="FullSimTest",
    )

    sim.run()

    # --- Assertions
    assert sim.t.yearvec[-1] == endyear
    assert sim.pars.n_agents == n_agents, "Simulation did not initialize with requested n_agents"
    assert len(sim.people) > 0
    alive_n = int(np.count_nonzero(sim.people.alive))
    assert alive_n > 0, "No living agents at end of simulation"
    assert alive_n <= len(sim.people), "Alive count cannot exceed total population"
    for analyzer in analyzers:
        assert hasattr(analyzer, "results"), f"{analyzer.name} missing results"

    print(" Full MIGHTI simulation ran successfully.")


if __name__ == "__main__":
    test_full_mighti_simulation()
