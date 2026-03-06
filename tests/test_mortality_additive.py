"""
Tests for additive-hazard mortality engine (background + modeled hazards).
"""

import numpy as np
import pandas as pd
import starsim as ss

import mighti as mi


class DummyPressure(ss.Disease):
    """A tiny module that reports constant death pressure for all alive agents."""

    def __init__(self, p, name="dummy"):
        super().__init__()
        self.name = name
        self._p = float(p)
        self._uids = np.array([], dtype=int)
        self._pp = np.array([], dtype=float)

    def init_pre(self, sim):
        super().init_pre(sim)
        self._uids = np.array([], dtype=int)
        self._pp = np.array([], dtype=float)

    def get_death_pressure(self):
        return self._uids, self._pp

    def step(self):
        # In "competing mortality protocol", the death engine sets sim._mighti_competing_mortality = True.
        ppl = self.sim.people
        auids = np.asarray(ppl.auids, dtype=int)
        self._uids = auids
        self._pp = np.full(len(auids), self._p, dtype=float)


def _simple_bg_rate():
    # Minimal long-format "mx schedule" compatible with mortality modules.
    # Use a constant mx across ages/sexes for one year.
    recs = []
    for sex in ("m", "f"):
        for a in range(0, 101):
            recs.append({"Time": 2007.0, "Sex": sex, "AgeGrpStart": a, "mx": 0.02})
    return pd.DataFrame(recs)


def _run_sim(with_pressure):
    bg = mi.mortality_additive.AdditiveHazardDeaths(
        _simple_bg_rate(),
        rate_units=1,
        background_multiplier=1.0,
    )
    diseases = [bg]
    if with_pressure:
        diseases.insert(0, DummyPressure(p=0.02, name="dummy"))

    mx_an = mi.analyzers.AgeSexMxAnalyzer(max_age=100)
    sim = ss.Sim(
        n_agents=2000,
        start=2007,
        stop=2008,
        diseases=diseases,
        analyzers=[mx_an],
        copy_inputs=False,
    )
    sim.run()
    e0 = mi.life_expectancy.calculate_life_expectancy_from_age_sex_mx_analyzer(sim, year=2008, max_age=100)
    return float(e0["Both"])


def test_additive_hazard_deaths_pressure_reduces_le():
    e0_no = _run_sim(with_pressure=False)
    e0_yes = _run_sim(with_pressure=True)
    assert e0_yes < e0_no, "Adding modeled hazard should reduce LE under additive hazards."

