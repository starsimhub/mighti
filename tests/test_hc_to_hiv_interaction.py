"""
Test that HC → HIV interactions can be represented via connectors.

We use HPV → HIV as an example, applying a relative susceptibility multiplier
to HIV acquisition risk among people currently infected with HPV.
"""

import numpy as np
import starsim as ss
import stisim as sti

import mighti as mi
from pathlib import Path


def test_hpv_increases_hiv_rel_sus():
    n_agents = 500
    thisdir = Path(__file__).resolve().parent
    param_path = str(thisdir / "test_data" / "eswatini_parameters.csv")

    # HIV needs a sexual network module available; we don't run transmission here,
    # but include the network for safe initialization.
    hiv = sti.HIV(init_prev=0.0, beta={"structuredsexual": [0.0, 0.0]})
    hpv = mi.diseases.HPV(
        csv_path=param_path,
        pars={"init_prev": ss.bernoulli(p=0.5)},
    )

    rr = 2.5
    conn = mi.interactions.create_dynamic_connector("HPV", "HIV", rr)

    sim = ss.Sim(
        n_agents=n_agents,
        start=2000,
        stop=2001,
        dt=1,
        networks=[sti.StructuredSexual()],
        diseases=[hiv, hpv],
        connectors=[conn],
        copy_inputs=False,
        label="hpv->hiv-rr-test",
    )

    sim.init()

    # Ensure baseline is 1.0 before applying connector logic
    sim.diseases.hiv.rel_sus[:] = 1.0

    hpv_inf = sim.diseases.hpv.infected.uids
    assert len(hpv_inf) > 0, "HPV should initialize with some infected agents"

    # Apply connector once
    conn.step()

    rel_sus = np.asarray(sim.diseases.hiv.rel_sus, dtype=float)
    assert np.allclose(rel_sus[hpv_inf], rr), "HIV rel_sus not increased for HPV-infected agents"

    not_hpv = np.setdiff1d(np.arange(n_agents), np.asarray(hpv_inf, dtype=int))
    if len(not_hpv):
        assert np.allclose(rel_sus[not_hpv], 1.0), "HIV rel_sus changed for agents without HPV"

