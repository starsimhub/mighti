import mighti as mi
import numpy as np
import sciris as sc
import starsim as ss
import pandas as pd

__all__ = ["SurvivorshipAnalyzer", "DeathAnalyzer"]
class SurvivorshipAnalyzer(ss.Analyzer):

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = 'survivorship_analyzer'

        self.max_age = max_age
        self.survivorship_data = {'Male': np.zeros(max_age), 'Female': np.zeros(max_age)}

    # def init_post(self):
    #     self.survivorship_data = np.zeros(shape=(self.max_age, 2))

    def step(self):
        ppl = self.sim.people
        for age in range(self.max_age):
            for sex in ['Male', 'Female']:
                self.survivorship_data[sex][age] += len(ppl.age[(ppl.age >= age) & (ppl.age < age+1) & (ppl.female == (sex=='Female'))])
       
                
class DeathAnalyzer(ss.Analyzer):
    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = 'death_analyzer'
        self.max_age = max_age
        
        # Initialize death counts per age and sex
        self.death_data = {'Male': np.zeros(max_age), 'Female': np.zeros(max_age)}

    def step(self):
        """Retrieve and store deaths for the current timestep."""
        ppl = self.sim.people
        current_timestep = self.sim.ti
        
        # Identify agents who died at this timestep
        deaths_this_step = ss.uids(np.where(ppl.ti_dead.raw == current_timestep)[0])  # Use ss.uids() for safe indexing

        # Update death data by age and sex
        for age in range(self.max_age):
            for sex in ['Male', 'Female']:
                # Filter deaths by age and sex
                deaths_in_group = deaths_this_step[
                    (ppl.age.raw[deaths_this_step] >= age) &
                    (ppl.age.raw[deaths_this_step] < age + 1) &
                    (ppl.female.raw[deaths_this_step] == (sex == 'Female'))
                ]
                self.death_data[sex][age] += len(deaths_in_group)

    def finalize(self):
        """Finalize the death data at the end of the simulation."""
        print(f"Final death data: {self.death_data}")