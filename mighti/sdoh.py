"""
Defines social determinants of health modules for housing, transportation, education, and income
"""


import starsim as ss
import numpy as np


__all__ = ['NeighbourhoodSituation', 'SocialContext', 'EducationSituation', 'EconomicSituation', 'HealthCareSystem']



import pandas as pd
import numpy as np
import starsim as ss

class NeighbourhoodSituation(ss.Module):
    """Models unstable housing as a binary state influenced by environment and inherits status from mother."""

    def __init__(self, csv_path=None, condition_name='NeighbourhoodSituation', default_p_stable=0.7, default_inherit_prob=0.9):
        super().__init__()
        self.name = 'neighbourhood'

        # Defaults
        self.p_stable = default_p_stable
        self.inherit_prob = default_inherit_prob

        # Load from CSV if provided
        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            row = df[df['state'].str.strip().str.lower() == condition_name.lower()]
            if not row.empty:
                if 'state_prob' in row:
                    self.p_stable = float(row['state_prob'].values[0])
                if 'inherit_prob' in row:
                    self.inherit_prob = float(row['inherit_prob'].values[0])

    def init_pre(self, sim):
        super().init_pre(sim)
        self.sim = sim
        self.state = sim.people.neighbourhood_situation

    def init_post(self):
        n = len(self.sim.people)
        vals = np.random.rand(n) < self.p_stable
        self.state[:] = vals

    def step(self):
        maternal = self.sim.networks['maternalnet']
        edges = maternal.edges

        new_birth_inds = np.where(edges.start == self.sim.ti)[0]
        if len(new_birth_inds) == 0:
            return

        mother_inds = edges.p1[new_birth_inds]
        baby_inds   = edges.p2[new_birth_inds]

        n = len(baby_inds)
        inherit_mask = np.random.rand(n) < self.inherit_prob

        # Inherit from mother
        self.state[baby_inds[inherit_mask]] = self.state[mother_inds[inherit_mask]]

        # Assign baseline to others
        random_vals = np.random.rand(n - inherit_mask.sum()) < self.p_stable
        self.state[baby_inds[~inherit_mask]] = random_vals


class SocialContext(ss.Module):
    def __init__(self, csv_path=None, condition_name='SocialContext', default_p_stable=0.6, default_inherit_prob=0.85):
        super().__init__()
        self.name = 'social'
        self.p_stable = default_p_stable
        self.inherit_prob = default_inherit_prob

        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            row = df[df['state'].str.strip().str.lower() == condition_name.lower()]
            if not row.empty:
                self.p_stable = float(row['state_prob'].values[0])
                self.inherit_prob = float(row['inherit_prob'].values[0])

    def init_pre(self, sim):
        super().init_pre(sim)
        self.sim = sim
        self.state = sim.people.social_context

    def init_post(self):
        n = len(self.sim.people)
        self.state[:] = np.random.rand(n) < self.p_stable

    def step(self):
        maternal = self.sim.networks['maternalnet']
        edges = maternal.edges
        new_inds = np.where(edges.start == self.sim.ti)[0]
        if len(new_inds) == 0: return
        moms, babies = edges.p1[new_inds], edges.p2[new_inds]
        mask = np.random.rand(len(babies)) < self.inherit_prob
        self.state[babies[mask]] = self.state[moms[mask]]
        self.state[babies[~mask]] = np.random.rand(len(babies) - mask.sum()) < self.p_stable


class EducationSituation(ss.Module):
    def __init__(self, csv_path=None, condition_name='EducationSituation', default_p_stable=0.75, default_inherit_prob=0.8):
        super().__init__()
        self.name = 'education'
        self.p_stable = default_p_stable
        self.inherit_prob = default_inherit_prob

        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            row = df[df['state'].str.strip().str.lower() == condition_name.lower()]
            if not row.empty:
                self.p_stable = float(row['state_prob'].values[0])
                self.inherit_prob = float(row['inherit_prob'].values[0])

    def init_pre(self, sim):
        super().init_pre(sim)
        self.sim = sim
        self.state = sim.people.education_situation

    def init_post(self):
        n = len(self.sim.people)
        self.state[:] = np.random.rand(n) < self.p_stable

    def step(self):
        maternal = self.sim.networks['maternalnet']
        edges = maternal.edges
        new_inds = np.where(edges.start == self.sim.ti)[0]
        if len(new_inds) == 0: return
        moms, babies = edges.p1[new_inds], edges.p2[new_inds]
        mask = np.random.rand(len(babies)) < self.inherit_prob
        self.state[babies[mask]] = self.state[moms[mask]]
        self.state[babies[~mask]] = np.random.rand(len(babies) - mask.sum()) < self.p_stable


class EconomicSituation(ss.Module):
    def __init__(self, csv_path=None, condition_name='EconomicSituation', default_p_stable=0.65, default_inherit_prob=0.9):
        super().__init__()
        self.name = 'economic'
        self.p_stable = default_p_stable
        self.inherit_prob = default_inherit_prob

        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            row = df[df['state'].str.strip().str.lower() == condition_name.lower()]
            if not row.empty:
                self.p_stable = float(row['state_prob'].values[0])
                self.inherit_prob = float(row['inherit_prob'].values[0])

    def init_pre(self, sim):
        super().init_pre(sim)
        self.sim = sim
        self.state = sim.people.economic_situation

    def init_post(self):
        n = len(self.sim.people)
        self.state[:] = np.random.rand(n) < self.p_stable

    def step(self):
        maternal = self.sim.networks['maternalnet']
        edges = maternal.edges
        new_inds = np.where(edges.start == self.sim.ti)[0]
        if len(new_inds) == 0: return
        moms, babies = edges.p1[new_inds], edges.p2[new_inds]
        mask = np.random.rand(len(babies)) < self.inherit_prob
        self.state[babies[mask]] = self.state[moms[mask]]
        self.state[babies[~mask]] = np.random.rand(len(babies) - mask.sum()) < self.p_stable
    
    
class HealthCareSystem(ss.Module):
    def __init__(self, csv_path=None, condition_name='HealthCareSystem', default_p_stable=0.6, default_inherit_prob=0.85):
        super().__init__()
        self.name = 'healthcare'
        self.p_stable = default_p_stable
        self.inherit_prob = default_inherit_prob

        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            row = df[df['state'].str.strip().str.lower() == condition_name.lower()]
            if not row.empty:
                self.p_stable = float(row['state_prob'].values[0])
                self.inherit_prob = float(row['inherit_prob'].values[0])

    def init_pre(self, sim):
        super().init_pre(sim)
        self.sim = sim
        self.state = sim.people.healthcare_system

    def init_post(self):
        n = len(self.sim.people)
        self.state[:] = np.random.rand(n) < self.p_stable

    def step(self):
        maternal = self.sim.networks['maternalnet']
        edges = maternal.edges
        new_inds = np.where(edges.start == self.sim.ti)[0]
        if len(new_inds) == 0: return
        moms, babies = edges.p1[new_inds], edges.p2[new_inds]
        mask = np.random.rand(len(babies)) < self.inherit_prob
        self.state[babies[mask]] = self.state[moms[mask]]
        self.state[babies[~mask]] = np.random.rand(len(babies) - mask.sum()) < self.p_stable
        