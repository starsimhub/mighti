"""
Defines interventions.
"""


import starsim as ss
import numpy as np


class ImproveHospitalDischarge(ss.Intervention):
    def __init__(self, disease_name, multiplier=2.0, start_day=0, end_day=None, label=None):
        super().__init__(label=label)
        self.disease_name = disease_name
        self.multiplier = multiplier
        self.start_day = start_day
        self.end_day = end_day

    def initialize(self, sim):
        self.sim = sim
        self.disease = sim.diseases[self.disease_name]

    def apply(self):
        ti = self.sim.ti
    
        # Always refresh the disease in case multiprocessing lost it
        if not hasattr(self, 'disease') or self.disease is None:
            try:
                self.disease = self.sim.diseases[self.disease_name]
                print(f"[{ti}] Reassigned self.disease to {self.disease_name}")
            except KeyError:
                raise ValueError(f"Disease '{self.disease_name}' not found. Available: {self.sim.diseases.keys()}")
    
        active = self.start_day <= ti < (self.end_day if self.end_day is not None else float('inf'))
    
        if active:
            self.disease.pars.p_daily_discharge_multiplier = self.multiplier
        else:
            self.disease.pars.p_daily_discharge_multiplier = 1.0
    
    def step(self):
        self.apply()
    

class GiveHousingToDepressed(ss.Intervention):
    def __init__(self, coverage=0.5, start_day=0, label=None):
        print(f"Intervention activated — changing housing for agents")

        super().__init__(label=label or "GiveHousingToDepressed")
        self.coverage = coverage
        self.start_day = start_day

    def initialize(self, sim):
        self.sim = sim  # Save reference to sim for use in step

    def apply(self):
        sim = self.sim
        if sim.ti < self.start_day:
            return

        # Safe search for depression
        depression = None
        for d in sim.diseases:
            if getattr(d, 'disease_name', '').lower() == 'depression':
                depression = d
                break

        if depression is None or not hasattr(depression, 'affected'):
            print(f"[{sim.ti}] Depression module not found or invalid")
            return

        housing_module = getattr(sim, 'housing_module', None)
        if housing_module is None or not hasattr(housing_module, 'housing_unstable'):
            print(f"[{sim.ti}] Housing module missing or housing_unstable not found")
            return

        depressed = depression.affected
        housing_unstable = housing_module.housing_unstable
        target = depressed & housing_unstable
        to_house = target[np.random.rand(len(target)) < self.coverage]
        housing_unstable[to_house] = False
        print(f"[{sim.ti}] Intervention activated — changed housing for {np.count_nonzero(to_house)} agents")
        
    def step(self):
        print(f"[{self.sim.ti}] step() called")

        self.apply()
        