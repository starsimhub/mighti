"""
Analyzers for demographic outcomes such as age-specific deaths and survivorship.
"""

import numpy as np
import pandas as pd
import starsim as ss


__all__ = ["DeathsByAgeSexAnalyzer", "SurvivorshipAnalyzer", "ConditionAtDeathAnalyzer"]

class DeathsByAgeSexAnalyzer(ss.Analyzer):
    """Tracks infant deaths and deaths by age/sex, Starsim-3.0.3 compatible."""

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        self._dead_prev = None  # snapshot of who was dead in previous step

    def init_pre(self, sim):
        super().init_pre(sim)
        self._dead_prev = np.zeros(len(sim.people), dtype=bool)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("infant_deaths", label="Cumulative infant deaths", dtype=int),
            ss.Result("male_deaths_by_age",   label="Male deaths by age",   dtype=int, shape=self.max_age + 1),
            ss.Result("female_deaths_by_age", label="Female deaths by age", dtype=int, shape=self.max_age + 1),
        )

    def _ensure_size(self):
        """Resize internal arrays if population size changes (future-proof)."""
        n_now = len(self.sim.people)
        if self._dead_prev is None:
            self._dead_prev = np.zeros(n_now, dtype=bool)
        elif self._dead_prev.size != n_now:
            new = np.zeros(n_now, dtype=bool)
            n_copy = min(self._dead_prev.size, n_now)
            new[:n_copy] = self._dead_prev[:n_copy]
            self._dead_prev = new

    def step(self):
        ppl = self.sim.people
        ti = self.sim.ti
        self._ensure_size()

        # New deaths this step
        new_deaths_mask = ppl.dead & ~self._dead_prev
        new_deaths = new_deaths_mask.uids

        # Cumulative infant deaths
        self.results.infant_deaths[ti] = int(np.count_nonzero(ppl.dead[ppl.age < 1]))

        # Tally new deaths by age/sex
        if len(new_deaths):
            ages = np.clip(np.floor(ppl.age[new_deaths]).astype(int), 0, self.max_age)
            fem = ppl.female[new_deaths]

            if np.any(fem):
                idx, cnt = np.unique(ages[fem], return_counts=True)
                self.results.female_deaths_by_age[idx] += cnt
            if np.any(~fem):
                idx, cnt = np.unique(ages[~fem], return_counts=True)
                self.results.male_deaths_by_age[idx] += cnt

        # Update snapshot
        self._dead_prev = np.array(ppl.dead, dtype=bool)

    def finalize(self):
        super().finalize() 

    def finalize_results(self):
        super().finalize_results()

    def to_df(self):
        ages = np.arange(self.max_age + 1)
        return pd.concat([
            pd.DataFrame({
                "age": ages, "sex": "Male",
                "deaths": self.results.male_deaths_by_age[:],
            }),
            pd.DataFrame({
                "age": ages, "sex": "Female",
                "deaths": self.results.female_deaths_by_age[:],
            }),
        ], ignore_index=True)


class SurvivorshipAnalyzer(ss.Analyzer):
    """
    Computes true survivorship l(x): fraction of original cohort surviving to age x.
    Compatible with all disease types (including neonatal).
    
    Note: Starsim removes dead people from the active population, so we can't use ppl.dead
    after the simulation. Instead, we track the initial population and use deaths from
    the DeathsByAgeSexAnalyzer to calculate survivorship.
    """

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = "survivorship_analyzer"
        self.max_age = max_age
        self._n0_male = None
        self._n0_female = None

    def init_pre(self, sim):
        """Store initial population counts by sex."""
        super().init_pre(sim)
        ppl = sim.people
        # Store initial population counts at birth (age 0)
        # We'll use the initial population size, not current size
        self._n0_male = max(np.sum(~ppl.female), 1)
        self._n0_female = max(np.sum(ppl.female), 1)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("lx_male",   shape=self.max_age + 1, dtype=float, label="Male survivorship l(x)"),
            ss.Result("lx_female", shape=self.max_age + 1, dtype=float, label="Female survivorship l(x)"),
        )

    def step(self):
        """No per-step logic (needed only to silence lifecycle warning)."""
        pass

    def finalize(self):
        """
        Set l(x) values for life table calculation.
        
        Note: For period life tables, l(x) is calculated from mx in calculate_life_table_from_mx.
        Here we just set l0 = 1.0 (radix) as the starting point. The actual l(x) values
        will be computed from mortality rates in the life table calculation.
        """
        super().finalize() 
        
        # For life table calculation, we just need l0 = 1.0 (radix)
        # The actual l(x) values will be calculated from mx in calculate_life_table_from_mx
        lx_m = np.ones(self.max_age + 1)  # Initialize to 1.0, will be recalculated from mx
        lx_f = np.ones(self.max_age + 1)  # Initialize to 1.0, will be recalculated from mx
        
        # Note: The actual l(x) values are not used in calculate_mortality_rates anymore
        # (we fixed that to use current population structure directly)
        # But we keep l0 = 1.0 for the life table calculation in calculate_life_table_from_mx

        self.results.lx_male[:] = lx_m
        self.results.lx_female[:] = lx_f

    def finalize_results(self):
        super().finalize_results()

    def to_df(self):
        records = []
        year = int(self.sim.t.yearvec[-1])
        for sex, key in (("Male", "lx_male"), ("Female", "lx_female")):
            lx = getattr(self.results, key)
            for age, val in enumerate(lx):
                records.append({
                    "year": year, "age": age, "sex": sex, "survival": float(val)
                })
        return pd.DataFrame(records)
    

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
        """Initialize analyzer results."""
        super().init_results()
        self.results = dict()
        self.results['n_deaths'] = 0
        self.results['by_cause'] = {}
        self.results['by_sex'] = {}
        self.results['by_age'] = {}

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
    