"""ClinicQueue: person-level slots, named rule, HTN treatment gated."""

import os

import numpy as np
import starsim as ss

from mighti.diseases.hypertension import Hypertension
from mighti.economics.clinic_queue import ClaimRecipe, ClinicQueue
from mighti.interventions.ncd_treatment import ChronicCareTx, ReduceMortalityTx


thisdir = os.path.dirname(__file__)
param_path = os.path.join(thisdir, "test_data", "eswatini_parameters.csv")


def _htn():
    return Hypertension(
        csv_path=param_path,
        pars={
            "init_prev": ss.bernoulli(0.8),
            "p_death": ss.bernoulli(0.0),
            "max_disease_duration": 10,
        },
    )


def _htn_demand(sim):
    htn = sim.diseases.hypertension
    n = int(sim.people.uid.len_used)
    return np.asarray(htn.affected.raw[:n], dtype=bool)


def _recipes():
    return {
        "followup": ClaimRecipe(
            "followup",
            eligibility=_htn_demand,
            slots=1.0,
            nurse_min=15.0,
        )
    }


def test_zero_slots_means_no_htn_treatment():
    queue = ClinicQueue(
        recipes=_recipes(),
        slots_per_step=0,
        nurse_min_per_step=0,
        rule="priority",
    )
    tx = ReduceMortalityTx(
        disease="hypertension",
        rel_death_reduction=0.5,
        service_claim="followup",
    )
    sim = ss.Sim(
        dict(
            start=2000,
            stop=2003,
            dt=1,
            n_agents=800,
            networks=[],
            diseases=[_htn()],
            interventions=[queue, tx],
        )
    ).run()
    n_on = int(np.asarray(sim.diseases.hypertension.on_treatment.raw, dtype=bool).sum())
    assert n_on == 0


def test_chronic_care_queue_claim_is_explicit():
    common = dict(start=2000, stop=2002, dt=1, n_agents=400, networks=[], rand_seed=1)
    sim_gated = ss.Sim(
        dict(
            **common,
            diseases=[_htn()],
            interventions=[
                ClinicQueue(
                    recipes=_recipes(),
                    slots_per_step=0,
                    nurse_min_per_step=0,
                ),
                ChronicCareTx(
                    disease="hypertension",
                    service_claim="followup",
                ),
            ],
        )
    ).run()
    sim_ungated = ss.Sim(
        dict(
            **common,
            diseases=[_htn()],
            interventions=[
                ClinicQueue(
                    recipes=_recipes(),
                    slots_per_step=0,
                    nurse_min_per_step=0,
                ),
                ChronicCareTx(disease="hypertension"),
            ],
        )
    ).run()
    assert len(sim_gated.diseases.hypertension.on_treatment.uids) == 0
    assert len(sim_ungated.diseases.hypertension.on_treatment.uids) > 0


def test_infinite_slots_treats_like_no_queue():
    queue = ClinicQueue(
        recipes=_recipes(),
        slots_per_step=np.inf,
        rule="first_come",
    )
    tx = ReduceMortalityTx(
        disease="hypertension",
        rel_death_reduction=0.5,
        service_claim="followup",
    )
    common = dict(start=2000, stop=2004, dt=1, n_agents=800, networks=[], rand_seed=1)
    sim_q = ss.Sim(
        dict(
            **common,
            diseases=[_htn()],
            interventions=[queue, tx],
        )
    ).run()
    sim_free = ss.Sim(
        dict(
            **common,
            diseases=[_htn()],
            interventions=[ReduceMortalityTx(disease="hypertension", rel_death_reduction=0.5)],
        )
    ).run()
    n_q = int(np.asarray(sim_q.diseases.hypertension.on_treatment.raw, dtype=bool).sum())
    n_free = int(np.asarray(sim_free.diseases.hypertension.on_treatment.raw, dtype=bool).sum())
    assert n_q > 0
    assert abs(n_q - n_free) <= max(5, int(0.05 * n_free))


def test_tight_slots_treat_fewer_than_infinite():
    tx = ReduceMortalityTx(
        disease="hypertension",
        rel_death_reduction=0.5,
        service_claim="followup",
    )
    common = dict(start=2000, stop=2004, dt=1, n_agents=800, networks=[], rand_seed=1)
    sim_tight = ss.Sim(
        dict(
            **common,
            diseases=[_htn()],
            interventions=[
                ClinicQueue(
                    recipes=_recipes(),
                    slots_per_step=5,
                    rule="priority",
                ),
                tx,
            ],
        )
    ).run()
    sim_free = ss.Sim(
        dict(
            **common,
            diseases=[_htn()],
            interventions=[
                ClinicQueue(
                    recipes=_recipes(),
                    slots_per_step=np.inf,
                    rule="priority",
                ),
                ReduceMortalityTx(
                    disease="hypertension",
                    rel_death_reduction=0.5,
                    service_claim="followup",
                ),
            ],
        )
    ).run()
    n_tight = int(np.asarray(sim_tight.diseases.hypertension.on_treatment.raw, dtype=bool).sum())
    n_free = int(np.asarray(sim_free.diseases.hypertension.on_treatment.raw, dtype=bool).sum())
    assert n_free > 0
    assert n_tight < n_free


def test_staff_time_does_not_roll_over():
    from mighti.economics.resource_accounting import ResourcePool

    pool = ResourcePool(budget_usd=100, hrh_minutes={"nurse": 10}, rollover_budget=True, rollover_hrh=False)
    pool.consume(cost=0, hrh_minutes={"nurse": 3}, source="visit")
    assert pool.remaining_hrh["nurse"] == 7
    pool.reset_rollover()
    assert pool.remaining_hrh["nurse"] == 10
    assert pool.remaining_budget == 100
