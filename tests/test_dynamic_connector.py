"""
Focused unit test for pairwise dynamic connectors created by
`mighti.interactions.create_dynamic_connector()`.

This test is intentionally minimal: it asserts that a connector updates the
target disease's `rel_sus` values for uids that are currently affected/infected
with the source disease.
"""

import os

import numpy as np
import starsim as ss
import mighti as mi


def test_dynamic_connector_applies_rel_sus_to_affected_uids():
    thisdir = os.path.dirname(__file__)
    param_path = os.path.join(thisdir, "test_data", "eswatini_parameters.csv")

    n_agents = 200
    ppl = ss.People(n_agents=n_agents)

    # Use two NCD-style diseases that expose `.affected` and `.rel_sus`
    source = mi.diseases.Type2Diabetes(
        csv_path=param_path,
        pars={"init_prev": ss.bernoulli(p=0.8)},
    )
    target = mi.diseases.AlcoholUseDisorder(
        csv_path=param_path,
        pars={"init_prev": ss.bernoulli(p=0.0)},
    )

    rel_sus_val = 3.0
    conn = mi.interactions.create_dynamic_connector("Type2Diabetes", "AlcoholUseDisorder", rel_sus_val)

    sim = ss.Sim(
        n_agents=n_agents,
        start=2000,
        stop=2001,
        people=ppl,
        diseases=[source, target],
        connectors=[conn],
        demographics=[],
        networks=[],
        copy_inputs=False,
        label="dynamic-connector-test",
    )

    sim.init()

    # Ensure we actually have some affected uids in the source
    src_uids = sim.diseases.type2diabetes.affected.uids
    assert len(src_uids) > 0, "Source disease has no affected uids; test is not meaningful"

    # Run a single connector step (no need to run the full sim)
    conn.step()

    rel_sus = np.asarray(sim.diseases.alcoholusedisorder.rel_sus, dtype=float)
    assert np.allclose(rel_sus[src_uids], rel_sus_val), "Target rel_sus not applied to affected uids"

    # Unaffected uids should remain at baseline (default=1.0)
    unaffected = np.setdiff1d(np.arange(n_agents), np.asarray(src_uids, dtype=int))
    if len(unaffected):
        assert np.allclose(rel_sus[unaffected], 1.0), "Target rel_sus changed for unaffected uids"

