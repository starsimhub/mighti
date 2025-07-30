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
    """
    Intervention that provides stable housing to individuals with Major Depressive Disorder
    who currently have unstable housing.
    """
    def __init__(self, coverage=0.5, start_day=0, label=None):
        super().__init__(label=label or "GiveHousingToDepressed")
        self.coverage = coverage
        self.start_day = start_day

    def initialize(self, sim):
        self.sim = sim

    def apply(self):
        sim = self.sim
        if sim.ti < self.start_day:
            return

        depression = sim.diseases.get('majordepressivedisorder', None)
        if depression is None or not hasattr(depression, 'affected'):
            print(f"[{sim.year}] MajorDepressiveDisorder not found or missing 'affected'")
            return

        # Target depressed + unstably housed
        ppl = sim.people
        depressed = depression.affected
        housing_unstable = ~ppl.neighbourhood_situation
        target = depressed & housing_unstable

        # Apply intervention with given coverage
        target_uids = target.uids
        n = len(target_uids)
        mask = np.random.rand(n) < self.coverage
        to_house = target_uids[mask]        
        ppl.neighbourhood_situation[to_house] = True

        
    def step(self):
        self.apply()
        