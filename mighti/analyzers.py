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
    def __init__(self, conditions=None, condition_attr_map=None, **kwargs):
        super().__init__(**kwargs)
        self.conditions = [c.lower() for c in (conditions or [])]
        self.condition_attr_map = condition_attr_map or {}
        self.records = []
        self.condition_snapshots = {}  # (uid, condition) → True/False

    def init_results(self):
        super().init_results()
        self.records = []
        self.condition_snapshots = {}

    def step(self):
        ppl = self.sim.people
        ti = self.sim.ti
        year = self.sim.t.yearvec[ti]

        for uid in ppl.dead.uids:
            record = {
                'uid': uid,
                'year': year,
                'age': ppl.age[uid],
                'sex': 'Female' if ppl.female[uid] else 'Male',
            }

            for cond in self.conditions:

                # if the condition has a different time step unit, adjust accordingly
                if not np.isnan(ppl[cond].ti_dead[uid]):
                    condition_ti = self.sim.diseases[cond].t.abstvec[int(ppl[cond].ti_dead[uid])]
                    record[f'died_{cond}'] = (condition_ti > ti-1) & (condition_ti <= ti)
                else:
                    record[f'died_{cond}'] = False

            self.records.append(record)

    def to_df(self):
        return pd.DataFrame(self.records)
    
    