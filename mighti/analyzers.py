import numpy as np
import starsim as ss

__all__ = ["DeathsByAgeSexAnalyzer"]

class DeathsByAgeSexAnalyzer(ss.Analyzer):

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('infant_deaths', label='Cumulative infant deaths', dtype=int),
            ss.Result('male_deaths_by_age', label='Number of male deaths by age', dtype=int, shape=101, ),
            ss.Result('female_deaths_by_age', label='Number of female deaths by age', dtype=int, shape=101, )
        )
        return


    def step(self):
        people = self.sim.people
        ti = self.sim.ti

        self.results.infant_deaths[ti] = len(people.dead[people.age < 1])

        for uid in people.dead.uids:
            age = int(min(people.age[uid], 100))            
            if people.female[uid]:
                self.results.female_deaths_by_age[age] += 1
            else:
                self.results.male_deaths_by_age[age] += 1