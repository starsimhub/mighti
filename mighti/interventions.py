"""
Defines interventions.
"""


import starsim as ss
import numpy as np

_all_ = ['ImproveHospitalDischarge', 'GiveHousingToDepressed', 'GiveHousingSupport', 'HousingSupportForAUD']

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


class GiveHousingSupport(ss.Intervention):
    def __init__(self, coverage=0.5, start_year=None, start_day=None, label=None):
        super().__init__(label=label or "GiveHousingSupport")
        self.coverage   = coverage
        self.start_year = start_year
        self.start_day  = start_day  # optional: still support direct ti

    def initialize(self, sim):
        self.sim = sim
        if self.start_day is None:
            if self.start_year is None:
                self.start_day = 0
            else:
                # Convert calendar year to ti
                self.start_day = max(0, int(round(self.start_year - sim.pars['start'])))
        # (optional) sanity print
        print(f"[Init] {self.label}: start_day={self.start_day}")

    def apply(self):
        sim = self.sim
        if sim.ti < self.start_day:
            return
        ppl = sim.people
        unstable = ~ppl.neighbourhood_situation
        adult = ppl.age >= 15
        target = unstable & adult
        uids = target.uids
        if len(uids):
            to_house = uids[np.random.rand(len(uids)) < self.coverage]
            ppl.neighbourhood_situation[to_house] = True
            print(f"[{sim.t.yearvec[sim.ti]:.1f}] {self.label} housed {len(to_house)} / {len(uids)}")
            # print(f"[{sim.year}] {self.label} housed {len(to_house)} / {len(uids)}")
    def step(self):
        self.apply()

class HousingSupportForAUD(ss.Intervention):
    """
    Provides supportive housing to adults with AUD who are unstably housed.
    Optionally reduces relapse risk after housing.
    """

    def __init__(self, coverage=0.5, start_year=2010, relapse_reduction=0.5, label=None):
        super().__init__(label=label or "HousingSupportForAUD")
        self.coverage = coverage
        self.start_year = start_year
        self.relapse_reduction = relapse_reduction

    def step(self):
        sim = self.sim
        current_year = sim.t.year
        if current_year < self.start_year:
            return

        ppl = sim.people
        aud = sim.diseases.alcoholusedisorder

        # Target: adults (≥15) who are unhoused and have AUD
        target = (~ppl.neighbourhood_situation) & aud.affected & (ppl.age >= 15)
        uids = target.uids
        if len(uids) == 0:
            return

        mask = np.random.rand(len(uids)) < self.coverage
        housed_uids = uids[mask]
        ppl.neighbourhood_situation[housed_uids] = True

        # Optional relapse protection
        if hasattr(aud, "relapse_rate"):
            aud.relapse_rate[housed_uids] *= self.relapse_reduction

        print(f"[{current_year:.1f}] {self.label} housed {len(housed_uids)} of {len(uids)} eligible adults with AUD")
        
        