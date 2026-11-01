"""
Tests for additive-hazard mortality (background + modeled causes).
"""

import numpy as np
import pandas as pd
import starsim as ss

import mighti as mi


class DummyCause(ss.Disease):
    """ A modeled cause of death: everyone alive dies of it at a constant probability per step, as a disease's deaths """

    def __init__(self, p, name="dummy"):
        super().__init__(name=name)
        self.define_pars(p_death=ss.bernoulli(p=p))

    def step(self):
        deaths = self.pars.p_death.filter(self.sim.people.auids)
        self.sim.people.request_death(deaths, cause=self)


def _simple_bg_rate():
    # Minimal long-format "mx schedule" compatible with mortality modules.
    # Use a constant mx across ages/sexes for one year.
    recs = []
    for sex in ("m", "f"):
        for a in range(0, 101):
            recs.append({"Time": 2007.0, "Sex": sex, "AgeGrpStart": a, "mx": 0.02})
    return pd.DataFrame(recs)


def _run_sim(with_cause):
    bg = mi.mortality_additive.AdditiveHazardDeaths(
        _simple_bg_rate(),
        rate_units=1,
        background_multiplier=1.0,
    )
    diseases = [DummyCause(p=0.02)] if with_cause else []

    mx_an = mi.analyzers.AgeSexMxAnalyzer(max_age=100)
    sim = ss.Sim(
        n_agents=2000,
        start=2007,
        stop=2008,
        demographics=bg,
        diseases=diseases,
        analyzers=[mx_an],
    )
    sim.run()
    e0 = mi.life_expectancy.calculate_life_expectancy_from_age_sex_mx_analyzer(sim, year=2008, max_age=100)
    return float(e0["Both"]), sim


def test_additive_hazard_deaths_pressure_reduces_le():
    e0_no, _ = _run_sim(with_cause=False)
    e0_yes, sim = _run_sim(with_cause=True)
    assert e0_yes < e0_no, "Adding modeled hazard should reduce LE under additive hazards."
    assert set(sim.people.causes) == {'background', 'dummy'}, "Each death should have one cause: the background or the modeled one"
