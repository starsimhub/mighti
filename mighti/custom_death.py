import numpy as np
import starsim as ss
from collections import defaultdict

class CustomDeaths(ss.Deaths):
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__(pars, metadata, **kwargs)
        self.cumulative_deaths = defaultdict(int)

    def step(self):
        """ Select people to die and track cumulative deaths """
        death_uids = self.pars.death_rate.filter()
        self.sim.people.request_death(death_uids)
        self.n_deaths = len(death_uids)
        # Track cumulative deaths by age
        for uid in death_uids:
            age = int(self.sim.people.age[uid])
            self.cumulative_deaths[age] += 1
        return self.n_deaths

    def get_cumulative_deaths(self):
        return dict(self.cumulative_deaths)

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        return