"""
Defines social determinants of health modules for housing, transportation, education, and income
"""


import starsim as ss
import numpy as np


__all__ = ['NeighbourhoodSituation', 'SocialContext', 'EducationSituation', 'EconomicSituation', 'HealthCareSystem']



class NeighbourhoodSituation(ss.Module):
    """Models unstable housing as a binary state influenced by employment."""

    def __init__(self, p_stable=0.7):
        super().__init__()
        self.name = 'neighbourhood'
        self.p_stable = p_stable

    def init_pre(self, sim):
        super().init_pre(sim)
        self.sim = sim
        self.state = sim.people.neighbourhood_situation

    def init_post(self):
        n = len(self.sim.people)
        vals = np.random.rand(n) < self.p_stable
        self.state[:] = vals

    def step(self):
        # Get mother-baby pairs from maternal network
        maternal = self.sim.networks['maternalnet']
        edges = maternal.edges

        # Select newborns whose connections started this timestep
        new_birth_inds = np.where(edges.start == self.sim.ti)[0]
        if len(new_birth_inds) == 0:
            return

        mother_inds = edges.p1[new_birth_inds]
        baby_inds   = edges.p2[new_birth_inds]

        # Inherit SDoH state from mothers (copied one timestep late)
        self.state[baby_inds] = self.state[mother_inds]



class SocialContext(ss.Connector):
    """Placeholder module for modeling access to SocialContext."""
    
    def step(self):
        pass


class EducationSituation(ss.Connector):
    """Placeholder module for modeling educational attainment."""

    def step(self):
        pass


class EconomicSituation(ss.Connector):
    """Placeholder module for modeling income level or poverty status."""
    
    def step(self):
        pass
    
    
class HealthCareSystem(ss.Connector):
    """Placeholder module for modeling healthcare system."""
    
    def step(self):
        pass
        