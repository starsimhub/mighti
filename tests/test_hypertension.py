"""Behavior tests for optional continuous-SBP hypertension physiology."""

import os

import numpy as np
import starsim as ss

from mighti.diseases.hypertension import Hypertension, SBPModel


thisdir = os.path.dirname(__file__)
param_path = os.path.join(thisdir, "test_data", "eswatini_parameters.csv")


def _sim(disease, *, stop=2002):
    return ss.Sim(
        dict(
            start=2000,
            stop=stop,
            dt=1,
            n_agents=200,
            networks=[],
            diseases=[disease],
            rand_seed=3,
        )
    ).run()


def test_binary_hypertension_remains_default():
    disease = Hypertension(
        csv_path=param_path,
        pars={"init_prev": ss.bernoulli(1.0), "p_death": ss.bernoulli(0.0)},
    )
    sim = _sim(disease, stop=2000)

    assert disease.uses_continuous_sbp is False
    assert len(sim.diseases.hypertension.affected.uids) == 200


def test_continuous_sbp_crossing_sets_states_and_results():
    disease = Hypertension(
        csv_path=param_path,
        pars={"init_prev": ss.bernoulli(1.0), "p_death": ss.bernoulli(0.0)},
        sbp_model=SBPModel(initial_sbp=100.0, annual_change=10.0, threshold=125.0),
    )
    sim = _sim(disease, stop=2002)
    htn = sim.diseases.hypertension

    assert htn.uses_continuous_sbp is True
    np.testing.assert_allclose(htn.sbp.raw[:200], 130.0)
    assert len(htn.raised_bp.uids) == 200
    assert len(htn.affected.uids) == 200
    assert int(np.sum(htn.results.new_cases)) == 200
    assert float(htn.results.mean_sbp[-1]) == 130.0
    assert float(htn.results.raised_bp_prevalence[-1]) == 1.0


def test_continuous_sbp_accepts_population_specific_callables():
    seen = {"initial": 0, "change": 0, "threshold": 0}

    def initial(sim, uids):
        seen["initial"] += len(uids)
        return np.where(uids % 2 == 0, 110.0, 130.0)

    def change(sim, uids):
        seen["change"] += len(uids)
        return np.zeros(len(uids))

    def threshold(sim, uids):
        seen["threshold"] += len(uids)
        return np.full(len(uids), 120.0)

    disease = Hypertension(
        csv_path=param_path,
        pars={"p_death": ss.bernoulli(0.0)},
        sbp_model=SBPModel(
            initial_sbp=initial,
            annual_change=change,
            threshold=threshold,
        ),
    )
    sim = _sim(disease, stop=2000)
    htn = sim.diseases.hypertension

    assert len(htn.affected.uids) == 100
    assert np.all(np.asarray(htn.raised_bp.raw[:200], dtype=bool)[1::2])
    assert not np.any(np.asarray(htn.raised_bp.raw[:200], dtype=bool)[::2])
    assert all(value > 0 for value in seen.values())


def test_continuous_sbp_requires_explicit_valid_specs():
    disease = Hypertension(
        csv_path=param_path,
        pars={"p_death": ss.bernoulli(0.0)},
        sbp_model=SBPModel(
            initial_sbp=np.array([110.0, 120.0]),
            annual_change=0.0,
            threshold=130.0,
        ),
    )

    try:
        _sim(disease, stop=2000)
    except ValueError as exc:
        assert "initial_sbp returned" in str(exc)
    else:
        raise AssertionError("Population-incompatible SBP input should fail")
