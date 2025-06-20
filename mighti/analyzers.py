"""
Analyzers for demographic outcomes such as age-specific deaths and survivorship.
"""

import numpy as np
import starsim as ss


__all__ = ["DeathsByAgeSexAnalyzer", "SurvivorshipAnalyzer"]


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
 