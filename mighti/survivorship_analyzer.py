import mighti as mi
import numpy as np
import sciris as sc
import starsim as ss
import pandas as pd

__all__ = ["SurvivorshipAnalyzer"]
class SurvivorshipAnalyzer(ss.Analyzer):

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = 'survivorship_analyzer'

        self.max_age = max_age
        self.survivorship_data = {'Male': np.zeros(max_age), 'Female': np.zeros(max_age)}

    # def init_results(self):
    #     n_timepoints = len(self.sim.t)
    #     self.survivorship_data = {
    #         'Male': np.zeros((self.max_age, n_timepoints)),
    #         'Female': np.zeros((self.max_age, n_timepoints)),
    #     }
        
    # def init_post(self):
    #     self.survivorship_data = np.zeros(shape=(self.max_age, 2))

    def step(self):
        ppl = self.sim.people
        ti = self.sim.ti
        # for age in range(self.max_age):
        #     for sex in ['Male', 'Female']:
        #         self.survivorship_data[sex][age] += len(ppl.age[(ppl.age >= age) & (ppl.age < age+1) & (ppl.female == (sex=='Female'))])
        for age in range(self.max_age):
            for sex in ['Male', 'Female']:
                count = len(ppl.age[(ppl.age >= age) & (ppl.age < age+1) & (ppl.female == (sex=='Female'))])
                self.survivorship_data[sex][age] = count