"""
Defines interventions.
"""


import starsim as ss
import stisim as sti
import numpy as np
import pandas as pd

_all_ = ['ART', 'ARTwithCASM', 'CustomART', 'ARTNoAutoAdjust', 'ImproveHospitalDischarge', 'GiveHousingToDepressed', 'GiveHousingSupport', 'HousingSupportForAUD']


# class ART(sti.ART):
#     """
#     ART intervention with optional integration to the BudgetConstraint module.
#     """

#     def init_pre(self, sim):
#         super().init_pre(sim)
#         # Store reference to budget module if present
#         self._budget_module = sim.get_module("budget_constraint", optional=True)

#     def apply(self, sim):
#         # Execute normal ART behavior (diagnosis, initiation, adherence updates, etc.)
#         super().apply(sim)

#         # If budget constraint active, register cost and HRH usage
#         if self._budget_module:
#             n_treated = getattr(self, "n_treated", 0)

#             # Safety guard: only proceed if n_treated > 0
#             if n_treated > 0:
#                 cost = n_treated * getattr(self, "cost_per_person_year", 120) / sim.n_years
#                 hrh_minutes = {
#                     "doctor": 5 * n_treated,
#                     "nurse": 30 * n_treated,
#                 }
#                 self._budget_module.register_usage(
#                     cost=cost,
#                     hrh_minutes=hrh_minutes,
#                     source=self.name,
#                 )


# class ARTwithCASM(sti.ART):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.casm_sensitivity = "pharma"


class CustomART(sti.ART):
    """
    ART intervention with:
      - Historical coverage as upper bound
      - Partial refill allowed via refill_factor
      - No art_coverage_correction (CASM-safe)
    """

    def __init__(self, pars=None, coverage_data=None, start_year=None,
                 refill_factor=1.0, **kwargs):

        super().__init__(pars=pars, coverage_data=coverage_data,
                         start_year=start_year, **kwargs)

        self.coverage_data = coverage_data
        self.refill_factor = float(refill_factor)

        # Determine if coverage is p_art or n_art
        if coverage_data is None:
            self.coverage_format = None
        else:
            if isinstance(coverage_data, pd.DataFrame):
                self.coverage_format = "p_art" if "p_art" in coverage_data.columns else None
            else:
                self.coverage_format = None

    def get_target_coverage(self, sim_year):
        """Return target proportion on ART for a given calendar year."""

        if self.coverage_data is None:
            return 0.0

        # DataFrame with year index and p_art column
        df = self.coverage_data

        if isinstance(df, pd.DataFrame):
            # Interpolate across years
            years = df.index.values.astype(float)
            vals = df.iloc[:,0].values.astype(float)
            return float(np.interp(sim_year, years, vals))

        raise ValueError("coverage_data must be a pandas DataFrame with year index.")

    def step(self):
        sim = self.sim
        hiv = sim.diseases.hiv

        # ------------------------------------------------
        # 1. Calendar year for this timestep
        # ------------------------------------------------
        sim_year = sim.t.now("year")

        # ------------------------------------------------
        # 2. Compute target proportion on ART
        # ------------------------------------------------
        if self.coverage_format == "p_art":
            p_target = self.get_target_coverage(sim_year)
        else:
            p_target = 0.0

        infected = hiv.infected.uids
        n_infected = len(infected)
        n_to_treat = int(p_target * n_infected)

        # ------------------------------------------------
        # 3. Apply natural ART stopping
        # ------------------------------------------------
        if hiv.on_art.any():
            stopping = hiv.on_art & (hiv.ti_stop_art <= self.ti)
            if stopping.any():
                hiv.stop_art(stopping.uids)

        # ------------------------------------------------
        # 4. Partial refill (CASM-safe)
        # ------------------------------------------------
        n_current = hiv.on_art.sum()
        raw_slots = max(n_to_treat - n_current, 0)
        n_slots = int(self.refill_factor * raw_slots)

        eligible = (hiv.diagnosed & ~hiv.on_art).uids
        n_slots = min(n_slots, len(eligible))

        if n_slots > 0:
            self.prioritize_art(sim, n=n_slots, awaiting_art_uids=eligible)

        # ------------------------------------------------
        # 5. DO NOT call art_coverage_correction
        # ------------------------------------------------

        # ------------------------------------------------
        # 6. MTCT protection
        # ------------------------------------------------
        if "pregnancy" in sim.demographics:
            preg = sim.people.pregnancy.pregnant
            if hiv.on_art[preg].any():
                mothers = (hiv.on_art & preg).uids
                infants = sim.networks.maternalnet.find_contacts(mothers)
                if len(infants):
                    hiv.rel_sus[ss.uids(infants)] = 0

        return


# Alias for backward compatibility
ARTNoAutoAdjust = CustomART


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
        
        