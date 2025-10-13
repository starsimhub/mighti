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
    """Tracks which conditions individuals had at the time of death and computes YLLs."""

    def __init__(self, conditions=None, condition_attr_map=None, ex_life_expectancy=80.0, **kwargs):
        super().__init__(**kwargs)
        self.conditions = [c.lower() for c in (conditions or [])]
        self.condition_attr_map = condition_attr_map or {}
        self.ex_life_expectancy = ex_life_expectancy
        self.records = []
        self.name = 'condition_at_death_analyzer'

    def init_results(self):
        self.records = []

    def step(self):
        ppl = self.sim.people
        ti = self.sim.t.ti
        current_year = self.sim.t.yearvec[ti]

        # Reference life expectancy (replace with country/sex-specific later if desired)
        life_expectancy_female = 75
        life_expectancy_male = 70

        # Loop over everyone who died this step
        for uid in ppl.dead.uids:
            age = ppl.age[uid]
            sex = 'Female' if ppl.female[uid] else 'Male'
            expected_le = life_expectancy_female if sex == 'Female' else life_expectancy_male
            yll = max(0, expected_le - age)

            record = dict(uid=uid, year=current_year, age=age, sex=sex, yll=yll)

            # For each condition, check if the person had it at death
            for cond in self.conditions:
                disease = getattr(self.sim.diseases, cond, None)
                if disease is None:
                    record[f'died_{cond}'] = False
                    continue

                # Choose attribute depending on disease type
                if hasattr(disease, 'infected'):
                    had_cond = disease.infected[uid]
                elif hasattr(disease, 'affected'):
                    had_cond = disease.affected[uid]
                elif hasattr(disease, 'active'):
                    had_cond = disease.active[uid]
                else:
                    had_cond = False

                record[f'died_{cond}'] = bool(had_cond)

            self.records.append(record)

    def to_df(self):
        """Return a DataFrame of all recorded deaths and associated conditions."""
        return pd.DataFrame(self.records)
    