"""Generic chronic care keeps people affected and does not compound effects."""

import os

import numpy as np
import pytest
import starsim as ss

from mighti.diseases.type2diabetes import Type2Diabetes
from mighti.interventions.ncd_treatment import ChronicCareTx, ReduceMortalityTx


thisdir = os.path.dirname(__file__)
param_path = os.path.join(thisdir, "test_data", "eswatini_parameters.csv")


def _t2d():
    return Type2Diabetes(
        csv_path=param_path,
        pars={
            "init_prev": ss.bernoulli(0.8),
            "p_death": ss.bernoulli(0.05),
            "max_disease_duration": 10,
        },
    )


def test_reduce_mortality_keeps_affected_and_does_not_compound():
    tx = ReduceMortalityTx(
        disease="type2diabetes",
        eligibility_mode="affected",
        rel_death_reduction=0.5,
        label="t2d_tx",
    )
    pars = dict(
        start=2000,
        stop=2010,
        dt=1,
        n_agents=2000,
        networks=[],
        diseases=[_t2d()],
        interventions=[tx],
    )
    sim = ss.Sim(pars).run()
    t2d = sim.diseases.type2diabetes
    on_tx = t2d.on_treatment.uids
    assert len(on_tx) > 0
    # Still have the condition
    assert bool((t2d.affected[on_tx]).all())
    # rel_death is 0.5, not 0.5 ** n_years
    rel = t2d.rel_death[on_tx]
    assert float(rel.min()) >= 0.49
    assert float(rel.max()) <= 0.51


def test_chronic_care_accepts_callable_eligibility():
    def even_uid_eligibility(sim, disease):
        n = int(sim.people.uid.len_used)
        return np.asarray(disease.affected.raw[:n], dtype=bool) & (np.arange(n) % 2 == 0)

    tx = ChronicCareTx(
        disease="type2diabetes",
        eligibility=even_uid_eligibility,
        rel_death_on_treatment=0.4,
    )
    sim = ss.Sim(
        dict(
            start=2000,
            stop=2002,
            dt=1,
            n_agents=1000,
            networks=[],
            diseases=[_t2d()],
            interventions=[tx],
        )
    ).run()
    t2d = sim.diseases.type2diabetes
    uids = np.asarray(t2d.on_treatment.uids, dtype=int)
    assert len(uids) > 0
    assert bool((uids % 2 == 0).all())
    assert bool((t2d.affected[uids]).all())
    assert np.allclose(np.asarray(t2d.rel_death[uids], dtype=float), 0.4)


def test_chronic_care_zero_initiation_probability_treats_nobody():
    tx = ChronicCareTx(
        disease="type2diabetes",
        initiation_prob_per_step=0.0,
    )
    sim = ss.Sim(
        dict(
            start=2000,
            stop=2002,
            dt=1,
            n_agents=500,
            networks=[],
            diseases=[_t2d()],
            interventions=[tx],
        )
    ).run()
    assert len(sim.diseases.type2diabetes.on_treatment.uids) == 0


def test_chronic_care_stopping_resets_owned_mortality_effect():
    class EligibleFirstStep:
        def __init__(self):
            self.calls = 0

        def __call__(self, sim, disease):
            self.calls += 1
            n = int(sim.people.uid.len_used)
            if self.calls == 1:
                return np.asarray(disease.affected.raw[:n], dtype=bool)
            return np.zeros(n, dtype=bool)

    tx = ChronicCareTx(
        disease="type2diabetes",
        eligibility=EligibleFirstStep(),
        rel_death_on_treatment=0.4,
    )
    sim = ss.Sim(
        dict(
            start=2000,
            stop=2001,
            dt=1,
            n_agents=500,
            networks=[],
            diseases=[_t2d()],
            interventions=[tx],
        )
    ).run()
    t2d = sim.diseases.type2diabetes
    assert len(t2d.on_treatment.uids) == 0
    affected = np.asarray(t2d.affected.raw, dtype=bool)
    assert np.allclose(np.asarray(t2d.rel_death.raw[affected], dtype=float), 1.0)


@pytest.mark.parametrize(
    "argument,value",
    [
        ("initiation_prob_per_step", -0.1),
        ("initiation_prob_per_step", 1.1),
        ("continuation_prob_per_step", -0.1),
        ("continuation_prob_per_step", 1.1),
    ],
)
def test_chronic_care_rejects_invalid_probabilities(argument, value):
    with pytest.raises(ValueError, match=argument):
        ChronicCareTx(disease="type2diabetes", **{argument: value})


def test_reduce_mortality_lowers_t2d_deaths():
    t2d0 = _t2d()
    t2d1 = _t2d()
    tx = ReduceMortalityTx(disease="type2diabetes", rel_death_reduction=0.5)
    sim0 = ss.Sim(
        dict(start=2000, stop=2020, dt=1, n_agents=4000, networks=[], diseases=[t2d0])
    ).run()
    sim1 = ss.Sim(
        dict(
            start=2000,
            stop=2020,
            dt=1,
            n_agents=4000,
            networks=[],
            diseases=[t2d1],
            interventions=[tx],
        )
    ).run()
    d0 = sim0.results.type2diabetes.new_deaths.values.sum()
    d1 = sim1.results.type2diabetes.new_deaths.values.sum()
    assert d1 < d0, f"T2D deaths should fall: {d1=} < {d0=}"
