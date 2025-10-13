"""
Test MIGHTI demography setup with people_extend.py
Ensures that extended population attributes (age, sex, etc.)
are correctly initialized and compatible with Starsim 3.x.
"""

import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import mighti as mi
import os


# Settings
n_agents = 500
inityear = 2007
do_plot = False
sc.options(interactive=do_plot)


def test_people_extend_basic():
    """
    Verify that make_people_with_age_sex() correctly
    initializes age–sex distribution and can attach to a simulation.
    """
    sc.heading("Testing people_extend initialization")

    thisdir = os.path.dirname(__file__)
    csv_path = os.path.join(thisdir, "test_data", "eswatini_age_distribution.csv")
    assert os.path.exists(csv_path), f"Missing test data: {csv_path}"

    ppl = mi.make_people_with_age_sex(csv_path=csv_path, init_year=inityear, n_agents=n_agents)

    assert isinstance(ppl, ss.People)
    assert len(ppl) == n_agents, f"Expected {n_agents}, got {len(ppl)}"
    assert hasattr(ppl.female, "default"), "Missing female default probability mapping"

    print(f"[✓] Created People with {n_agents} agents using people_extend.py")
    return ppl


def test_people_extend_with_demography():
    """
    Attach the People object to a minimal simulation with Deaths and Pregnancy.
    This triggers Starsim to initialize and populate all arrays.
    """
    sc.heading("Testing people_extend compatibility with Demography")

    ppl = test_people_extend_basic()

    thisdir = os.path.dirname(__file__)
    death_csv = os.path.join(thisdir, "test_data", "eswatini_mortality_rates.csv")
    fert_csv = os.path.join(thisdir, "test_data", "eswatini_asfr.csv")
    death_rates = {"death_rate": pd.read_csv(death_csv), "rate_units": 1}
    fertility_rate = {"fertility_rate": pd.read_csv(fert_csv)}

    death = ss.Deaths(death_rates)
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    sim = ss.Sim(
        people=ppl,
        demographics=[pregnancy, death],
        start=2007,
        stop=2009,
        n_agents=n_agents,
        copy_inputs=False,
        label="DemographyTest",
    )
    sim.run()

    # Starsim has now populated arrays
    assert hasattr(sim.people, "age"), "Missing age array after init"
    assert hasattr(sim.people, "female"), "Missing female attribute after init"

    # Age distribution should have realistic range
    assert 0 <= np.min(sim.people.age) <= 5
    assert np.max(sim.people.age) < 100

    # Check female array structure
    female_arr = sim.people.female.bool if hasattr(sim.people.female, "bool") else sim.people.female[:]
    assert isinstance(female_arr, np.ndarray)
    assert female_arr.dtype == bool

    # Check basic mortality tracking
    total_deaths = len(sim.people.dead)
    assert total_deaths >= 0
    assert hasattr(sim.results, "new_deaths")
    print(f"[✓] Simulation completed; total deaths recorded: {total_deaths}")
    print("[✓] people_extend integration test passed.")


if __name__ == "__main__":
    test_people_extend_basic()
    test_people_extend_with_demography()
    print("\n[✓] All demography tests passed successfully.")
    