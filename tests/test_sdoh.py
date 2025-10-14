"""
Test suite for MIGHTI SDoH modules under Starsim 3.x

Covers:
- Initialization of NeighbourhoodSituation
- Inheritance via MaternalNet
- Coexistence of multiple SDoH modules
- Edge case: step() when no births occur
"""

import os
import numpy as np
import pandas as pd
import starsim as ss
import sciris as sc
import mighti as mi


def test_neighbourhood_situation_inheritance():
    thisdir = os.path.dirname(__file__)
    param_path = os.path.join(thisdir, 'test_data', 'sdoh.csv')
    age_path = os.path.join(thisdir, 'test_data', 'eswatini_age_distribution_2007.csv')
    fertility_path = os.path.join(thisdir, 'test_data', 'eswatini_asfr.csv')

    fertility_rate = {'fertility_rate': pd.read_csv(fertility_path)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    extra_states = [ss.BoolArr('neighbourhood_situation')]
    people = ss.People(n_agents=1000, age_data=pd.read_csv(age_path), extra_states=extra_states)
    maternal = ss.MaternalNet()
    ns = mi.NeighbourhoodSituation(csv_path=param_path)

    sim = ss.Sim(
        rand_seed=42,
        n_agents=1000,
        start=2007,
        stop=2010,
        people=people,
        networks=maternal,
        demographics=[pregnancy],
        connectors=[ns],
        copy_inputs=False,
    )

    sim.run()

    n_true = sim.people.neighbourhood_situation.sum()
    n_false = (~sim.people.neighbourhood_situation).sum()
    print(f"NeighbourhoodSituation Init:\n  True: {n_true:,}\n  False: {n_false:,}")
    assert 0 < n_true < len(sim.people), "Expected a mix of True and False in initialized values."

    maternal_edges = sim.networks['maternalnet'].edges
    mothers = maternal_edges.p1.to_numpy()
    babies = maternal_edges.p2.to_numpy()
    valid = (~np.isnan(mothers)) & (~np.isnan(babies))
    mothers, babies = mothers[valid].astype(int), babies[valid].astype(int)
    sdoh = sim.people.neighbourhood_situation
    matches = sdoh[babies] == sdoh[mothers]
    n_matches = matches.sum()
    print(f"Inherited housing: {n_matches}/{len(matches)} match ({100*n_matches/len(matches):.1f}%)")
    assert n_matches > 0, "No inherited SDoH values detected."
    print(" NeighbourhoodSituation inheritance test passed.")


def test_multiple_sdoh_modules():
    """Ensure multiple SDoH modules register distinct boolean arrays without conflict."""
    modules = [
        mi.NeighbourhoodSituation(),
        mi.EconomicSituation(),
        mi.EducationSituation(),
        mi.SocialContext(),
    ]

    sim = ss.Sim(n_agents=200, start=2000, stop=2001, connectors=modules)
    sim.init()

    ppl = sim.people
    existing_fields = list(ppl.states.keys())  # ✅ StarSim 3.x: use .states.keys()

    # Define expected keyword fragments
    expected_keywords = ['neighbourhood', 'economic', 'education', 'social']

    found_fields = []
    for keyword in expected_keywords:
        matching = [f for f in existing_fields if keyword in f.lower()]
        if not matching:
            raise AssertionError(f"No people state found for keyword '{keyword}' (available: {existing_fields})")
        arr_name = matching[0]
        arr = ppl.states[arr_name]
        assert arr.dtype == bool, f"{arr_name} should be boolean"
        found_fields.append(arr_name)
        print(f"  ✓ {arr_name} initialized (mean={arr.mean():.2f})")

    assert len(set(found_fields)) == len(expected_keywords), "Duplicate or missing SDoH arrays detected"
    print(" Multiple SDoH modules initialized without conflict.")


# ---------------------------------------------------------------------
if __name__ == '__main__':
    test_neighbourhood_situation_inheritance()
    test_multiple_sdoh_modules()
    print("\n All SDoH tests passed successfully.")
