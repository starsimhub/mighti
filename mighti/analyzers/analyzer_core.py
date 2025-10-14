"""
Analyzers for demographic outcomes such as age-specific deaths and survivorship.
"""

import numpy as np
import pandas as pd
import starsim as ss


__all__ = ["DeathsByAgeSexAnalyzer", "SurvivorshipAnalyzer", "ConditionAtDeathAnalyzer"]


class DeathsByAgeSexAnalyzer(ss.Analyzer):
    """Tracks infant deaths and age- and sex-specific deaths."""

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('infant_deaths', label='Cumulative infant deaths', dtype=int),
            ss.Result('male_deaths_by_age', label='Number of male deaths by age', dtype=int, shape=101),
            ss.Result('female_deaths_by_age', label='Number of female deaths by age', dtype=int, shape=101)
        )

    def step(self):
        people = self.sim.people
        ti = self.sim.ti

        self.results.infant_deaths[ti] = len(people.dead[people.age < 1])

        for uid in people.dead.uids:
            age_capped = min(int(np.floor(people.age[uid])), 100)
            if people.female[uid]:
                self.results.female_deaths_by_age[age_capped] += 1
            else:
                self.results.male_deaths_by_age[age_capped] += 1


class SurvivorshipAnalyzer(ss.Analyzer):
    """Computes survivorship by age and sex for life table construction."""

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = 'survivorship_analyzer'

        self.max_age = max_age
        self.survivorship_data = {'Male': np.zeros(max_age), 'Female': np.zeros(max_age)}

    def step(self):
        ppl = self.sim.people
        for age in range(self.max_age):
            for sex in ['Male', 'Female']:
                self.survivorship_data[sex][age] += len(ppl.age[(ppl.age >= age) & (ppl.age < age+1) & (ppl.female == (sex=='Female'))])


class ConditionAtDeathAnalyzer(ss.Analyzer):
    """
    Simple analyzer that records who died, their age/sex/YLL,
    and which non-HIV conditions they had or died from.

    Notes:
    - HIV deaths are excluded (use the HIV module directly).
    - Analyzer still records HIV-positive status at death.
    - No cause grouping logic here (handled in main scripts).
    """

    def __init__(self, conditions=None, ex_life_expectancy=80.0, **kwargs):
        super().__init__(**kwargs)
        # Store non-HIV conditions only (HIV tracked separately)
        self.conditions = [c.lower() for c in (conditions or []) if c.lower() != "hiv"]
        self.ex_life_expectancy = ex_life_expectancy
        self.records = []
        self.name = "condition_at_death_analyzer"

    def init_results(self):
        self.records = []

    def _had_condition(self, disease, uid):
        """Return True if agent had this condition at death."""
        if disease is None:
            return False
        for attr in ("infected", "affected", "active"):
            if hasattr(disease, attr):
                arr = getattr(disease, attr)
                try:
                    return bool(arr[uid])
                except Exception:
                    return False
        return False

    def _died_of_condition(self, disease, uid, ti):
        """Return True if this module has ti_dead and matches current step."""
        if disease is None or not hasattr(disease, "ti_dead"):
            return False
        ti_dead = getattr(disease, "ti_dead")
        try:
            return bool((ti_dead == ti)[uid])
        except Exception:
            raw = getattr(ti_dead, "raw", None)
            if raw is None:
                return False
            return bool(np.isfinite(raw[uid]) and raw[uid] == ti)

    def step(self):
        ppl = self.sim.people
        ti = self.sim.t.ti
        year = float(self.sim.t.yearvec[ti])
        died_now = ppl.dead.uids
        if not len(died_now):
            return

        # Reference HIV module to check HIV infection at death
        hiv_mod = getattr(self.sim.diseases, "hiv", None)

        for uid in died_now:
            age = float(ppl.age[uid])
            sex = "Female" if ppl.female[uid] else "Male"

            # Simple YLL estimate (replace later with life-table lookup if needed)
            le = 75 if sex == "Female" else 70
            if isinstance(self.ex_life_expectancy, (int, float)):
                le = float(self.ex_life_expectancy)
            yll = max(0.0, le - age)

            rec = dict(uid=int(uid), year=year, age=age, sex=sex, yll=yll)

            # Record presence and cause flags for each disease
            for cond in self.conditions:
                disease = getattr(self.sim.diseases, cond, None)
                rec[f"died_{cond}"] = self._had_condition(disease, uid)
                rec[f"cause_{cond}"] = self._died_of_condition(disease, uid, ti)

            # Record HIV infection status (not death cause)
            rec["hiv_positive"] = False
            if hiv_mod is not None and hasattr(hiv_mod, "infected"):
                try:
                    rec["hiv_positive"] = bool(hiv_mod.infected[uid])
                except Exception:
                    pass

            # Determine primary cause (non-HIV only)
            causes = [c for c in self.conditions if rec.get(f"cause_{c}", False)]
            if len(causes) == 1:
                rec["primary_cause"] = causes[0]
            elif len(causes) > 1:
                rec["primary_cause"] = "multiple"
            else:
                rec["primary_cause"] = None

            self.records.append(rec)

    def to_df(self):
        """Return DataFrame of recorded deaths and conditions."""
        return pd.DataFrame(self.records)
    