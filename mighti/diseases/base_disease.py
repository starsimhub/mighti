"""
Defines health conditions and their base logic, including disease-specific behavior and initialization.
"""

import logging
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import lognorm

from mighti.util.rng import get_rng


__all__ = ['RemittingDisease', 'AcuteDisease', 'AcuteSurgicalDisease', 'ChronicDisease',
            'GenericSIS', 'GenericSIR', 'NonAcquiredDisease', 'StaticCondition']


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

def get_disease_parameters(csv_path, disease_name):
    """
    Load disease-specific parameters from a CSV file, returning a dictionary
    with required fields and defaults when missing.

    Parameters:
        csv_path (str): Path to the parameter CSV file.
        disease_name (str): Name of the disease to look up.

    Returns:
        dict: Dictionary of parameters for the specified disease.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "condition" not in df.columns:
        raise KeyError(f"'condition' column missing in {csv_path}. Available columns: {df.columns.tolist()}")

    row = df[df["condition"] == disease_name]
    if row.empty:
        raise ValueError(f"Disease '{disease_name}' not found in parameter file: {csv_path}")

    def get_value_safe(field, default):
        if field not in row.columns:
            logger.warning(f"Column '{field}' missing for {disease_name}, using default: {default}")
            return default
        val = row[field].values[0]
        if pd.isna(val):
            logger.warning(f"Missing value for '{field}' in {disease_name}, using default: {default}")
            return default
        return val

    return {
        "p_death": get_value_safe("p_death", 0.0001),
        "dur_condition": get_value_safe("dur_condition", 10),
        "rel_sus_hiv": get_value_safe("rel_sus", 1.0),
        "remission_rate": get_value_safe("remission_rate", 0.0),
        "max_disease_duration": get_value_safe("max_disease_duration", 30),
        "affected_sex": get_value_safe("affected_sex", "both"),
        "p_acquire": get_value_safe("p_acquire", 0.01),
        "p_acquire_male": get_value_safe("p_acquire_male", get_value_safe("p_acquire", 0.01)),
        "p_acquire_female": get_value_safe("p_acquire_female", get_value_safe("p_acquire", 0.01)),
    }


class _CompetingMortalityMixin:
    """
    Minimal protocol for "competing mortality" mode.

    - Disease modules should *report* death pressure via `_set_death_pressure()`
      instead of directly calling `people.request_death()`.
    - `mighti.mortality_competing.CompetingRisksDeaths` will allocate all-cause
      deaths and store an attribution mapping on the sim for this timestep.
    - During `step_die()`, we set `ti_dead` (and optionally results) only for
      deaths attributed to this module.
    """

    def _competing_enabled(self):
        sim = getattr(self, "sim", None)
        return bool(getattr(sim, "_mighti_competing_mortality", False))

    def _set_death_pressure(self, uids, p):
        self._death_pressure_uids = np.asarray(uids, dtype=int)
        self._death_pressure_p = np.asarray(p, dtype=float)

    def get_death_pressure(self):
        u = getattr(self, "_death_pressure_uids", None)
        p = getattr(self, "_death_pressure_p", None)
        if u is None or p is None:
            return np.array([], dtype=int), np.array([], dtype=float)
        return u, p

    def _attributed_deaths(self, death_uids):
        sim = getattr(self, "sim", None)
        cause_map = getattr(sim, "_mighti_death_cause", None)
        if not isinstance(cause_map, dict) or not len(death_uids):
            return np.array([], dtype=int)
        my_name = getattr(self, "name", self.__class__.__name__)
        keep = [uid for uid in death_uids if cause_map.get(int(uid)) == my_name]
        return np.asarray(keep, dtype=int)

    def step_die(self, uids):
        # In legacy mode, deaths (and ti_dead) are handled inside step()
        if not self._competing_enabled():
            return

        ti = getattr(self, "ti", None)
        if ti is None:
            return

        attributed = self._attributed_deaths(np.asarray(uids, dtype=int))
        if len(attributed):
            if hasattr(self, "ti_dead"):
                self.ti_dead[attributed] = ti
            if hasattr(self, "results") and "new_deaths" in getattr(self, "results", {}):
                try:
                    self.results.new_deaths[ti] = len(attributed)
                except Exception:
                    pass
        return


class RemittingDisease(_CompetingMortalityMixin, ss.NCD):
    """ Base class for all remitting diseases."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path    
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)        
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        # Calculate the mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        # Define parameters using extracted values
        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
            p_death=ss.bernoulli(disease_params["p_death"]),  
            remission_rate=disease_params["remission_rate"],  
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire_multiplier_male=1.0,
            p_acquire_multiplier_female=1.0,
            p_acquire=disease_params["p_acquire"],
            p_acquire_male=disease_params["p_acquire_male"],
            p_acquire_female=disease_params["p_acquire_female"],
            # Avoid changing distribution types by setting None (Starsim restriction).
            init_prev=ss.bernoulli(0.0),
        )
        
        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate) 

        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),   
            ss.BoolState('affected'),
            ss.BoolState('on_treatment'),
            ss.BoolState('reversed'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'), 
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),  
            reset=True,
        )

    def init_post(self):

        super().init_post()

        # (1) initialize baseline risk if relevant
        initial_risk = self.pars['initial_risk'].filter()
        self.at_risk[initial_risk] = True
        self.ti_affected[initial_risk] = self.ti + self.pars['dur_risk'].rvs(initial_risk, round=True)

        # (2) initialize prevalence
        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            probs = self.pars.init_prev.rvs(self.sim.people.uid)          # ← fixed
            rng = get_rng(self.sim, salt=f"init_prev:{getattr(self, 'disease_name', self.__class__.__name__)}")
            affected = rng.random(len(self.sim.people)) < probs

            if hasattr(self, "affected"):
                self.affected[:] = affected

            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True

    def init_results(self):
        super().init_results()
        existing_results = set(self.results.keys())

        if 'new_cases' not in existing_results:
            self.define_results(ss.Result('new_cases', dtype=int, label='New Cases'))
        if 'new_deaths' not in existing_results:
            self.define_results(ss.Result('new_deaths', dtype=int, label='Deaths'))
        if 'prevalence' not in existing_results:
            self.define_results(ss.Result('prevalence', dtype=float, label='Prevalence'))
        if 'remission_prevalence' not in existing_results:
            self.define_results(ss.Result('remission_prevalence', dtype=float, label='Remission Prevalence'))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

    def step_state(self):
        if hasattr(self, "p_remission"):
            going_into_remission = self.p_remission.filter(self.affected.uids) 
            self.affected[going_into_remission] = False
            self.reversed[going_into_remission] = True
            self.ti_reversed[going_into_remission] = self.ti

            recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
            self.reversed[recovered] = False
            self.susceptible[recovered] = True  

    def step(self):
        ti = self.ti

        susceptible = (~self.affected).uids
        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        draws = rng.random(len(susceptible))
        new_cases = susceptible[draws < p_acq]

        self.affected[new_cases] = True
        self.ti_affected[new_cases] = ti

        # Dynamic death logic — allows rel_death to be changed over time
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]

        try:
            base_p = self.pars.p_death.pars['p']  # extract base death prob
        except Exception:
            raise ValueError(f"Cannot extract base death probability from {self.pars.p_death}")

        adjusted_p_death = base_p * rel_death
        if self._competing_enabled():
            # Report death pressure; actual deaths are allocated by CompetingRisksDeaths
            self._set_death_pressure(affected_uids, adjusted_p_death)
            deaths = np.array([], dtype=int)
            self.results.new_deaths[ti] = 0
        else:
            draws = rng.random(len(affected_uids))
            deaths = affected_uids[draws < adjusted_p_death]
            self.ti_dead[deaths] = ti
            self.sim.people.request_death(deaths)
            self.results.new_deaths[ti] = len(deaths)

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        return new_cases

    @property
    def duration(self):
        """Duration of active condition in years, with NaN-safety."""
        if not hasattr(self, 'ti_affected') or not hasattr(self, 'affected'):
            raise AttributeError("This disease does not support duration")

        n = len(self.sim.people)
        dur = np.zeros(n)
        ti_now = self.ti

        # Defensive copy and clean any nan or invalid times
        ti_aff = np.asarray(self.ti_affected, dtype=float)
        ti_aff[~np.isfinite(ti_aff)] = 0.0

        # active indices that exist within current population size
        active = self.affected.uids[self.affected.uids < n]
        if len(active):
            dur[active] = np.maximum(0, ti_now - ti_aff[active])

        # Replace any remaining NaN with 0
        dur[~np.isfinite(dur)] = 0.0
        return dur


class AcuteDisease(_CompetingMortalityMixin, ss.NCD):
    """Base class for all acute diseases."""

    def __init__(self, csv_path=None, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        # Calculate mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire_multiplier_male=1.0,
            p_acquire_multiplier_female=1.0,
            p_acquire=disease_params["p_acquire"],
            p_acquire_male=disease_params["p_acquire_male"],
            p_acquire_female=disease_params["p_acquire_female"],
            # Avoid Starsim "update dist to NoneType" errors.
            init_prev=ss.bernoulli(0.0),
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),
            ss.BoolState('affected'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            reset=True,
        )

    def init_post(self):
        
        super().init_post()
        sim = self.sim

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            # Sample prevalence probabilities
            probs = self.pars.init_prev.rvs(sim.people.uid)
            rng = get_rng(sim, salt=f"{self.__class__.__name__}:init_prev")
            affected = rng.random(len(sim.people)) < probs

            # Assign disease state
            if hasattr(self, "affected"):
                self.affected[:] = affected

            # Optionally set prognoses for affected agents
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True
        self.at_risk[uids] = False

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases', int, 'New Cases'),
            ('new_deaths', int, 'Deaths'),
            ('prevalence', float, 'Prevalence')
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        susceptible = self.at_risk.uids
        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        p_death = base_p * rel_death
        if self._competing_enabled():
            self._set_death_pressure(affected_uids, p_death)
            deaths = np.array([], dtype=int)
        else:
            deaths = affected_uids[rng.random(len(affected_uids)) < p_death]
            self.sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return new_cases


    @property
    def duration(self):
        """
        Duration (in years) since onset of disease, 0 if not affected.
        This allows YLD calculations in MicrocostingAnalyzer.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)

        # Handle different onset attributes
        if hasattr(self, 'affected') and hasattr(self, 'ti_affected'):
            affected_uids = self.affected.uids
            if len(affected_uids):
                dur[affected_uids] = self.sim.t.years - self.ti_affected[affected_uids]
        elif hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            infected_uids = self.infected.uids
            if len(infected_uids):
                dur[infected_uids] = self.sim.t.years - self.ti_infected[infected_uids]

        # Clip negatives (e.g. from pre-sim infections)
        dur = np.clip(dur, 0, None)
        return dur
    


class AcuteSurgicalDisease(_CompetingMortalityMixin, ss.NCD):
    """
    Acute disease with a possible surgical intervention event.

    Represents conditions like appendicitis, congenital heart anomalies, or digestive congenital anomalies
    that are acute in course but can be surgically treated to improve survival.

    Parameters loaded from CSV include:
        - dur_condition: mean duration of untreated disease (yrs)
        - p_death: baseline probability of death per timestep
        - p_acquire: per-timestep acquisition probability
        - p_surgery: probability of receiving surgery
        - rel_mortality_treated: relative mortality for treated individuals
        - rel_mortality_untreated: relative mortality for untreated individuals
        - cost_surgery (optional): for MicrocostingAnalyzer integration
    """

    def __init__(self, csv_path=None, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params.get("rel_sus_hiv", 1.0),
            affected_sex=disease_params.get("affected_sex", "both"),
            p_acquire_multiplier=1.0,
            p_acquire_multiplier_male=1.0,
            p_acquire_multiplier_female=1.0,
            p_acquire=disease_params["p_acquire"],
            p_acquire_male=disease_params["p_acquire_male"],
            p_acquire_female=disease_params["p_acquire_female"],
            p_surgery=disease_params.get("p_surgery", 0.3),
            rel_mortality_treated=disease_params.get("rel_mortality_treated", 0.5),
            rel_mortality_untreated=disease_params.get("rel_mortality_untreated", 2.0),
            cost_surgery=disease_params.get("cost_surgery", 0.0),
            init_prev=ss.bernoulli(0.0),
        )

        self.p_acquire = ss.bernoulli(
            p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids)
        )
        self.update_pars(pars, **kwargs)

        # Define states
        self.define_states(
            ss.BoolState("susceptible", default=True),
            ss.BoolState("at_risk", default=True),
            ss.BoolState("affected"),
            ss.BoolState("on_treatment"),   # here: has received surgery
            ss.BoolState("surgery_done", default=False),
            ss.FloatArr("ti_affected"),
            ss.FloatArr("ti_dead"),
            ss.FloatArr("ti_surgery"),
            ss.FloatArr("rel_sus", default=1.0),
            ss.FloatArr("rel_death", default=1.0),
            reset=True,
        )

    def init_post(self):
        super().init_post()
        sim = self.sim

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            probs = self.pars.init_prev.rvs(sim.people.uid)
            rng = get_rng(sim, salt=f"{self.__class__.__name__}:init_prev")
            affected = rng.random(len(sim.people)) < probs
            if hasattr(self, "affected"):
                self.affected[:] = affected
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])
        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True
        self.at_risk[uids] = False
        self.rel_death[uids] = self.pars.rel_mortality_untreated

    def init_results(self):
        super().init_results()
        existing_results = set(self.results.keys())

        if "new_cases" not in existing_results:
            self.define_results(ss.Result("new_cases", dtype=int, label="New Cases"))
        if "new_deaths" not in existing_results:
            self.define_results(ss.Result("new_deaths", dtype=int, label="Deaths"))
        if "new_surgeries" not in existing_results:
            self.define_results(ss.Result("new_surgeries", dtype=int, label="Surgeries"))
        if "prevalence" not in existing_results:
            self.define_results(ss.Result("prevalence", dtype=float, label="Prevalence"))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        sim = self.sim
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")

        # --- Acquisition ---
        susceptible = self.at_risk.uids
        p_acq = calculate_p_acquire_generic(self, sim, susceptible)

        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti
        self.rel_death[new_cases] = self.pars.rel_mortality_untreated

        # --- Surgery events ---
        affected_uids = self.affected.uids
        can_surgery = affected_uids[~self.surgery_done[affected_uids]]
        surgeries = can_surgery[rng.random(len(can_surgery)) < self.pars.p_surgery]
        if len(surgeries):
            self.on_treatment[surgeries] = True
            self.surgery_done[surgeries] = True
            self.ti_surgery[surgeries] = ti
            self.rel_death[surgeries] = self.pars.rel_mortality_treated

        # --- Deaths ---
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get("p", 0)
        p_death = base_p * rel_death
        if self._competing_enabled():
            self._set_death_pressure(affected_uids, p_death)
            deaths = np.array([], dtype=int)
        else:
            deaths = affected_uids[rng.random(len(affected_uids)) < p_death]
            if len(deaths):
                sim.people.request_death(deaths)
                self.ti_dead[deaths] = ti

        # --- Results ---
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_surgeries[ti] = len(surgeries)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)

        return new_cases

    @property
    def duration(self):
        """Duration (in years) since onset of disease."""
        n = len(self.sim.people)
        dur = np.zeros(n)
        if hasattr(self, "affected") and hasattr(self, "ti_affected"):
            affected_uids = self.affected.uids
            if len(affected_uids):
                dur[affected_uids] = self.sim.t.years - self.ti_affected[affected_uids]
        dur = np.clip(dur, 0, None)
        return dur
    

class ChronicDisease(_CompetingMortalityMixin, ss.NCD):
    """Base class for chronic diseases."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire_multiplier_male=1.0,
            p_acquire_multiplier_female=1.0,
            p_acquire=disease_params["p_acquire"],
            p_acquire_male=disease_params["p_acquire_male"],
            p_acquire_female=disease_params["p_acquire_female"],
            init_prev=ss.bernoulli(0.0),
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),
            ss.BoolState('affected'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            reset=True,
        )

    def init_post(self):
 
        super().init_post()
        sim = self.sim 

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            # Sample prevalence probabilities
            probs = self.pars.init_prev.rvs(sim.people.uid)
            rng = get_rng(sim, salt=f"{self.__class__.__name__}:init_prev")
            affected = rng.random(len(sim.people)) < probs

            # Assign disease state
            if hasattr(self, "affected"):
                self.affected[:] = affected

            # Optionally set prognoses for affected agents
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True
        self.at_risk[uids] = False

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases', int, 'New Cases'),
            ('new_deaths', int, 'Deaths'),
            ('prevalence', float, 'Prevalence')
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        susceptible = self.at_risk.uids
        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        p_death = base_p * rel_death
        if self._competing_enabled():
            self._set_death_pressure(affected_uids, p_death)
            deaths = np.array([], dtype=int)
        else:
            deaths = affected_uids[rng.random(len(affected_uids)) < p_death]
            self.sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return new_cases

    @property
    def duration(self):
        """
        Duration (in years) since onset of disease, 0 if not affected.
        This allows YLD calculations in MicrocostingAnalyzer.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)

        # Handle different onset attributes
        if hasattr(self, 'affected') and hasattr(self, 'ti_affected'):
            affected_uids = self.affected.uids
            if len(affected_uids):
                dur[affected_uids] = self.sim.t.years - self.ti_affected[affected_uids]
        elif hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            infected_uids = self.infected.uids
            if len(infected_uids):
                dur[infected_uids] = self.sim.t.years - self.ti_infected[infected_uids]

        # Clip negatives (e.g. from pre-sim infections)
        dur = np.clip(dur, 0, None)
        return dur
    

class GenericSIS(_CompetingMortalityMixin, ss.SIS):
    """Base class for communicable diseases (SIS model)."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            remission_rate=disease_params["remission_rate"],
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire_multiplier_male=1.0,
            p_acquire_multiplier_female=1.0,
            p_acquire=disease_params["p_acquire"],
            p_acquire_male=disease_params["p_acquire_male"],
            p_acquire_female=disease_params["p_acquire_female"],
            init_prev=pars.get("init_prev", ss.bernoulli(0)) if pars else ss.bernoulli(0),
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate)
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),
            ss.BoolState('infected'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_infected'),
            # Starsim's SIS implementation expects `ti_recovered` for recovery timing/state updates.
            # We keep `ti_reversed` for backward-compatibility with older MIGHTI codepaths.
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            reset=True,
        )

    def step_state(self):
        """
        Handle remission/recovery transitions (I → S).

        Notes
        -----
        Starsim's built-in `ss.SIS.step_state()` relies on `ti_recovered` being defined and
        scheduled. MIGHTI historically modeled remission as a per-timestep probability
        (`remission_rate`), so we implement recovery directly here for robustness across
        Starsim versions.
        """
        ti = self.ti
        infected_uids = self.infected.uids
        if not len(infected_uids):
            return

        if hasattr(self, "p_remission"):
            rec = infected_uids[self.p_remission.filter(infected_uids)]
        else:
            # Fallback: treat remission_rate as per-timestep probability
            p = float(getattr(self.pars, "remission_rate", 0.0) or 0.0)
            if p <= 0.0:
                return
            rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step_state")
            rec = infected_uids[rng.random(len(infected_uids)) < p]

        if not len(rec):
            return

        self.infected[rec] = False
        self.susceptible[rec] = True
        self.at_risk[rec] = True

        # Track recovery time for analyzers and for any downstream code expecting it
        if hasattr(self, "ti_recovered"):
            self.ti_recovered[rec] = ti
        if hasattr(self, "ti_reversed"):
            self.ti_reversed[rec] = ti

    def init_post(self):
        """
        Initialize infection prevalence.

        We intentionally do NOT call Starsim's `Infection.init_post()` here because it uses
        `init_prev.filter()` without passing uids, which is fragile for callable p(uids).
        """
        # Still call the base Module hook to initialize state arrays and satisfy Starsim's
        # required-method checker.
        ss.Module.init_post(self)

        sim = self.sim

        init_prev = getattr(self.pars, "init_prev", None)
        if init_prev is None:
            return

        # Prefer Starsim distribution semantics: filter(uids) returns infected uids
        if callable(getattr(init_prev, "filter", None)):
            try:
                initial_uids = init_prev.filter(sim.people.uid)
            except TypeError:
                # Fallback for Starsim distributions that take no args
                initial_uids = init_prev.filter()
            if len(initial_uids):
                self.set_prognoses(initial_uids, sources=-1)
            try:
                self.pars._n_initial_cases = len(initial_uids)
            except Exception:
                pass
        return

    def set_prognoses(self, uids, sources=None, **kwargs):  # noqa: ARG002
        """Set prognoses upon infection (Starsim-compatible signature)."""
        # Do not call ss.SIS.set_prognoses() since we do not define SIS immunity states.
        # Only call the base Disease logger hook.
        try:
            ss.Disease.set_prognoses(self, uids, sources)
        except Exception:
            pass
        ti = self.t.ti
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.at_risk[uids] = False
        if hasattr(self, "ti_infected"):
            self.ti_infected[uids] = ti

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases', int, 'New Cases'),
            ('new_deaths', int, 'Deaths'),
            ('prevalence', float, 'Prevalence')
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        susceptible = self.at_risk.uids & self.susceptible.uids
        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        self.infected[new_cases] = True
        self.susceptible[new_cases] = False
        self.at_risk[new_cases] = False
        self.ti_infected[new_cases] = ti

        # Deaths
        affected_uids = self.infected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        p_death = base_p * rel_death
        if self._competing_enabled():
            self._set_death_pressure(affected_uids, p_death)
            deaths = np.array([], dtype=int)
        else:
            deaths = affected_uids[rng.random(len(affected_uids)) < p_death]
            self.sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)
        return new_cases
    
    @property
    def duration(self):
        """
        Duration (in years) since onset of disease, 0 if not affected.
        This allows YLD calculations in MicrocostingAnalyzer.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)

        # Handle different onset attributes
        if hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            affected_uids = self.infected.uids
            if len(affected_uids):
                dur[affected_uids] = self.sim.t.years - self.ti_infected[affected_uids]
        elif hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            infected_uids = self.infected.uids
            if len(infected_uids):
                dur[infected_uids] = self.sim.t.years - self.ti_infected[infected_uids]

        # Clip negatives (e.g. from pre-sim infections)
        dur = np.clip(dur, 0, None)
        return dur


class GenericSIR(_CompetingMortalityMixin, ss.SIR):
    """Base class for communicable diseases (SIR model)."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)
        
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            remission_rate=disease_params["remission_rate"],   # per-timestep recovery prob
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire_multiplier_male=1.0,
            p_acquire_multiplier_female=1.0,
            p_acquire=disease_params["p_acquire"],             # force of infection term
            p_acquire_male=disease_params["p_acquire_male"],
            p_acquire_female=disease_params["p_acquire_female"],
            init_prev=pars.get("init_prev", ss.bernoulli(0)) if pars else ss.bernoulli(0),
        )

        # Stochastic processes
        self.p_acquire   = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate)

        self.update_pars(pars, **kwargs)

        # States
        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),     # convenience mask for who can acquire
            ss.BoolState('infected'),
            ss.BoolState('recovered'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus',   default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            reset=True,
        )

    def init_post(self):
        """
        Initialize infection prevalence.

        Do not call Starsim's base init_post; see GenericSIS.init_post rationale.
        """
        # Still call the base Module hook to initialize state arrays and satisfy Starsim's
        # required-method checker.
        ss.Module.init_post(self)

        sim = self.sim

        init_prev = getattr(self.pars, "init_prev", None)
        if init_prev is None:
            return

        if callable(getattr(init_prev, "filter", None)):
            try:
                initial_uids = init_prev.filter(sim.people.uid)
            except TypeError:
                initial_uids = init_prev.filter()
            if len(initial_uids):
                self.set_prognoses(initial_uids, sources=-1)
            try:
                self.pars._n_initial_cases = len(initial_uids)
            except Exception:
                pass
        return

    def set_prognoses(self, uids, sources=None, **kwargs):  # noqa: ARG002
        """Set prognoses upon infection (Starsim-compatible signature)."""
        # Avoid ss.SIR.set_prognoses(); we manage event timing ourselves.
        try:
            ss.Disease.set_prognoses(self, uids, sources)
        except Exception:
            pass
        ti = self.t.ti
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.at_risk[uids] = False
        if hasattr(self, "ti_infected"):
            self.ti_infected[uids] = ti

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases',   int,   'New Cases'),
            ('new_deaths',  int,   'Deaths'),
            ('prevalence',  float, 'Prevalence (Infected)'),
            ('recovered',   int,   'New Recoveries'),
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        sim = self.sim
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")

        # --- Acquire infection (S → I) ---
        susceptible = self.at_risk.uids & self.susceptible.uids  # ensure truly in S
        p_acq = calculate_p_acquire_generic(self, sim, susceptible)

        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        if len(new_cases):
            self.infected[new_cases]    = True
            self.susceptible[new_cases] = False
            self.at_risk[new_cases]     = False
            self.ti_infected[new_cases] = ti

        # --- Recoveries (I → R, no reinfection in classic SIR) ---
        infected_uids = self.infected.uids
        new_rec = infected_uids[self.p_remission.filter(infected_uids)]
        if len(new_rec):
            self.infected[new_rec]  = False
            self.recovered[new_rec] = True
            self.ti_recovered[new_rec] = ti

        # --- Deaths among infected (optional relative risk) ---
        rel_death = self.rel_death[infected_uids] if len(infected_uids) else np.array([])
        base_p = self.pars.p_death.pars.get('p', 0)
        p_death = base_p * (rel_death if len(rel_death) else 1.0)
        if self._competing_enabled():
            self._set_death_pressure(infected_uids, p_death)
            deaths = np.array([], dtype=int)
        else:
            deaths = infected_uids[rng.random(len(infected_uids)) < p_death]
            if len(deaths):
                sim.people.request_death(deaths)
                self.ti_dead[deaths] = ti

        # --- Results ---
        self.results.new_cases[ti]  = len(new_cases)
        self.results.recovered[ti]  = len(new_rec)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.infected) / len(sim.people)

        return new_cases

    @property
    def duration(self):
        """
        Duration (years) since infection onset for currently infected;
        recovered or susceptible return 0. Useful for YLD calculations.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)
        if hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            iu = self.infected.uids
            if len(iu):
                dur[iu] = self.sim.t.years - self.ti_infected[iu]
        dur = np.clip(dur, 0, None)
        return dur
    
    
class NonAcquiredDisease(_CompetingMortalityMixin, ss.Disease):
    """
    Base class for congenital or neonatal (non-acquired) diseases.

    Used for:
        - Neonatal conditions (encephalopathy, preterm birth, sepsis)
        - Congenital anomalies (heart, limb, digestive)
        - Static genetic disorders (Down Syndrome, Chromosomal Abnormalities)

    Features:
        - No acquisition or remission processes
        - No 'at_risk' or 'susceptible' states
        - Static prevalence initialized at birth
        - Optional neonatal restriction (<28 days)
        - Mortality via p_death
    """
    depends_on = ["Deaths", "DeathsExtended"]

    def __init__(self, csv_path, pars=None, is_neonatal=False, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        self.is_neonatal = is_neonatal
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        # Load parameters
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)

        # Define parameters (no acquisition or remission)
        self.define_pars(
            p_death=ss.bernoulli(disease_params.get("p_death", 0.0)),
            dur_condition=disease_params.get("dur_condition", 1.0),
            max_disease_duration=disease_params.get("max_disease_duration", 1.0),
            rel_sus_hiv=disease_params.get("rel_sus_hiv", 1.0),
            affected_sex=disease_params.get("affected_sex", "both"),
            init_prev=pars.get("init_prev", ss.bernoulli(0.01)) if pars else ss.bernoulli(0.01),
        )
        self.update_pars(pars, **kwargs)

        # Define minimal states
        self.define_states(
            ss.BoolState("affected", default=False, label="Affected"),
            ss.FloatArr("rel_death", default=1.0, label="Relative mortality multiplier"),
            ss.FloatArr("rel_sus", default=1.0, label="Relative susceptibility"),
            ss.FloatArr("ti_affected", label="Time of becoming affected"),
            ss.FloatArr("ti_dead", label="Time of death"),
            reset=True,
        )

    # ---------------------------------------------------------------------
    # Initialization lifecycle
    # ---------------------------------------------------------------------
    def init_pre(self, sim):
        super().init_pre(sim)
        return

    def init_post(self):
        """Initialize congenital/neonatal prevalence at birth."""
        super().init_post()
        sim = self.sim
        n = len(sim.people)
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:init_prev")

        # For neonatal diseases: only initialize among existing neonates at sim start.
        # For congenital/static conditions: initialize across the whole population at sim start
        # to represent prevalent conditions in the initial age structure.
        if self.is_neonatal:
            ages = getattr(sim.people, "age_years", sim.people.age)
            target_uids = np.where(np.asarray(ages, dtype=float) < (28 / 365))[0].astype(int)
        else:
            target_uids = np.arange(n, dtype=int)

        affected = np.zeros(n, dtype=bool)
        if len(target_uids):
            affected[target_uids] = self._draw_affected_for_uids(target_uids, rng=rng)

        self.affected[:] = affected
        self.ti_affected[affected] = self.ti

        n_affected = affected.sum()
        logger.info(f"[INIT] {self.disease_name}: {n_affected}/{n} ({n_affected/n:.3%}) affected at birth")

    def _draw_affected_for_uids(self, uids, *, rng):
        """
        Draw boolean affected status for a given uid subset using pars.init_prev.

        Supports:
        - StarSim distributions with .rvs(uids)
        - Callables returning bool arrays
        - Scalar float probabilities
        """
        uids = np.asarray(uids, dtype=int)

        init_prev = getattr(self.pars, "init_prev", None)
        if init_prev is None:
            return np.zeros(len(uids), dtype=bool)

        # StarSim distribution-like
        if hasattr(init_prev, "rvs"):
            try:
                draws = init_prev.rvs(uids)
            except Exception:
                draws = init_prev.rvs(self.sim.people.uid)
                draws = np.asarray(draws)[uids]
            draws = np.asarray(draws)
            if draws.dtype == bool:
                return draws.astype(bool)
            # interpret as probabilities
            probs = np.asarray(draws, dtype=float)
            probs = np.clip(probs, 0.0, 1.0)
            return rng.random(len(uids)) < probs

        # Callable returning bool/probabilities
        if callable(init_prev):
            draws = np.asarray(init_prev(), dtype=float)
            if draws.dtype == bool:
                return draws[uids].astype(bool) if draws.size == len(self.sim.people) else draws.astype(bool)
            if draws.size == len(self.sim.people):
                probs = np.clip(draws[uids], 0.0, 1.0)
            else:
                probs = np.clip(draws, 0.0, 1.0)
            return rng.random(len(uids)) < probs

        # Scalar probability
        p = float(init_prev)
        p = float(np.clip(p, 0.0, 1.0))
        return rng.random(len(uids)) < p

    def _newborn_uids_this_step(self):
        """Return newborn uids for this timestep using MaternalNet edges, if present."""
        sim = self.sim
        maternal = sim.networks.get("maternalnet", None) if hasattr(sim, "networks") else None
        if maternal is None or not hasattr(maternal, "edges"):
            return np.array([], dtype=int)

        edges = maternal.edges
        if not hasattr(edges, "start") or not hasattr(edges, "p2"):
            return np.array([], dtype=int)

        try:
            birth_inds = np.where(np.asarray(edges.start) == sim.ti)[0]
        except Exception:
            return np.array([], dtype=int)

        if birth_inds.size == 0:
            return np.array([], dtype=int)

        babies = np.asarray(edges.p2)[birth_inds]
        babies = babies[np.isfinite(babies)].astype(int, copy=False)
        babies = babies[(babies >= 0) & (babies < len(sim.people))]
        if babies.size == 0:
            return np.array([], dtype=int)

        # Deduplicate in case of repeated edges
        return np.unique(babies)

    def _assign_newborn_cases(self):
        """Assign affected status to newborns at the current timestep."""
        babies = self._newborn_uids_this_step()
        if babies.size == 0:
            return

        sim = self.sim
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:birth")
        affected_babies = self._draw_affected_for_uids(babies, rng=rng)

        if affected_babies.size != babies.size:
            affected_babies = np.resize(np.asarray(affected_babies, dtype=bool), babies.size)

        # Write state
        self.affected[babies] = affected_babies
        self.ti_affected[babies[affected_babies]] = self.ti

    def init_results(self):
        super().init_results()
        existing = set(self.results.keys())
        if "prevalence" not in existing:
            self.define_results(ss.Result("prevalence", dtype=float, scale=False, label="Prevalence"))
        if "new_deaths" not in existing:
            self.define_results(ss.Result("new_deaths", dtype=int, scale=True, label="Deaths"))
        if "n_affected" not in existing:
            self.define_results(ss.Result("n_affected", dtype=int, scale=False, label="Affected"))

    # ---------------------------------------------------------------------
    # Step logic
    # ---------------------------------------------------------------------
    def step_state(self):
        """No within-step state transitions for congenital diseases."""
        return

    def step(self):
        """Apply mortality among affected individuals."""
        # Always assign newborn cases first (if births occurred this timestep)
        self._assign_newborn_cases()
        # Ensure no stale pressure is carried across timesteps
        if self._competing_enabled():
            self._set_death_pressure(np.array([], dtype=int), np.array([], dtype=float))

        # Skip the very first timestep so deaths are not applied before survivorship baseline
        if self.ti == 0:
            return

        ti = self.ti
        sim = self.sim
        affected_uids = self.affected.uids
        if not len(affected_uids):
            return

        # Restrict to neonates if needed
        if self.is_neonatal:
            ages = getattr(sim.people, "age_years", sim.people.age)
            affected_uids = affected_uids[ages[affected_uids] < (28 / 365)]
            if not len(affected_uids):
                return

        base_p = self.pars.p_death.pars.get("p", 0)
        rel_death = self.rel_death[affected_uids]
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")
        p_death = base_p * rel_death
        if self._competing_enabled():
            self._set_death_pressure(affected_uids, p_death)
            deaths = np.array([], dtype=int)
            self.results.new_deaths[ti] = 0
        else:
            deaths = affected_uids[rng.random(len(affected_uids)) < p_death]
            if len(deaths):
                sim.people.request_death(deaths)
                logger.debug(f"[STEP] {self.disease_name}: {len(deaths)} deaths at timestep {ti}")
            self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        self.results.n_affected[ti] = np.count_nonzero(self.affected)

    def step_die(self, uids):
        """Record cause-attributed death times for this condition."""
        uids = np.asarray(uids, dtype=int)
        if not len(uids):
            return

        if self._competing_enabled():
            attributed = self._attributed_deaths(uids)
            if len(attributed):
                self.ti_dead[attributed] = self.sim.people.ti_dead[attributed]
                # new_deaths is set here so downstream analyzers see it even if step() wrote 0
                try:
                    self.results.new_deaths[self.ti] = len(attributed)
                except Exception:
                    pass
            return

        # Legacy behavior: record death times for *anyone affected at death* (used by some analyses)
        affected_dead = uids[self.affected[uids]]
        if len(affected_dead):
            self.ti_dead[affected_dead] = self.sim.people.ti_dead[affected_dead]
        return

    # ---------------------------------------------------------------------
    # Results tracking
    # ---------------------------------------------------------------------
    def update_results(self):
        super().update_results()
        ti = self.ti
        sim = self.sim
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        self.results.n_affected[ti] = np.count_nonzero(self.affected)

    # ---------------------------------------------------------------------
    # Finalization lifecycle
    # ---------------------------------------------------------------------
    def finalize(self):
        super().finalize()
        if self._competing_enabled():
            # Do not overwrite per-timestep cause-attributed `ti_dead`
            return
        ppl = self.sim.people
        dead = ppl.dead.uids
        affected_dead = dead[self.affected[dead]]
        if len(affected_dead):
            self.ti_dead[affected_dead] = ppl.ti_dead[affected_dead]
        logger.debug(f"[FINAL] {self.disease_name}: {len(affected_dead)} total deaths recorded.")

    def finalize_results(self):
        super().finalize_results()

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def duration(self):
        """Duration (years) of condition presence since birth."""
        n = len(self.sim.people)
        dur = np.zeros(n)
        affected_uids = self.affected.uids
        if len(affected_uids):
            current_time = self.sim.t.years
            dur[affected_uids] = current_time - self.ti_affected[affected_uids]
        return dur
    

class StaticCondition(NonAcquiredDisease):
    """
    Lifelong static (non-progressive) conditions like Down Syndrome or chromosomal abnormalities.
    """

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__(csv_path, pars, is_neonatal=False, **kwargs)
        self.define_pars(dur_condition=np.inf, max_disease_duration=np.inf)

    def step(self):
        """Lifelong condition — mortality only, no remission."""
        ti = self.ti
        sim = self.sim

        # Assign to newborns if births occurred this timestep
        self._assign_newborn_cases()
        if self._competing_enabled():
            self._set_death_pressure(np.array([], dtype=int), np.array([], dtype=float))

        # Skip the very first timestep for consistency with NonAcquiredDisease
        if self.ti == 0:
            return np.array([])

        affected = self.affected.uids
        if not len(affected):
            return np.array([])

        rel_death = self.rel_death[affected]
        base_p = self.pars.p_death.pars.get("p", 0)
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")
        p_death = base_p * rel_death
        if self._competing_enabled():
            self._set_death_pressure(affected, p_death)
            deaths = np.array([], dtype=int)
        else:
            deaths = affected[rng.random(len(affected)) < p_death]
            if len(deaths):
                sim.people.request_death(deaths)
                self.ti_dead[deaths] = ti

        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        return deaths    

def calculate_p_acquire_generic(disease, sim, uids):
    """Calculate acquisition probability for a disease with optional sex filtering and HIV interaction."""
    p_acquire = float(getattr(disease.pars, "p_acquire", 1.0))
    p_acquire_male = float(getattr(disease.pars, "p_acquire_male", p_acquire))
    p_acquire_female = float(getattr(disease.pars, "p_acquire_female", p_acquire))
    default_mult = float(getattr(disease.pars, "p_acquire_multiplier", 1.0))
    male_mult = float(getattr(disease.pars, "p_acquire_multiplier_male", default_mult))
    female_mult = float(getattr(disease.pars, "p_acquire_multiplier_female", default_mult))
    p_base = np.full(len(uids), default_mult * p_acquire, dtype=float)

    try:
        male_mask = np.asarray(sim.people.male[uids], dtype=bool)
        female_mask = np.asarray(sim.people.female[uids], dtype=bool)
    except Exception:
        male_mask = None
        female_mask = None

    affected = str(getattr(disease.pars, "affected_sex", "both")).strip().lower()
    if affected == "female":
        if female_mask is not None:
            p_base[female_mask] = female_mult * p_acquire_female
        if male_mask is not None:
            p_base[male_mask] = 0.0
    elif affected == "male":
        if male_mask is not None:
            p_base[male_mask] = male_mult * p_acquire_male
        if female_mask is not None:
            p_base[female_mask] = 0.0
    else:
        if male_mask is not None:
            p_base[male_mask] = male_mult * p_acquire_male
        if female_mask is not None:
            p_base[female_mask] = female_mult * p_acquire_female

    try:
        if hasattr(sim.people, 'hiv'):
            hiv_positive = sim.people.hiv[uids]
            p_base[hiv_positive] *= disease.pars.rel_sus_hiv
    except Exception:
        pass

    try:
        return p_base * disease.rel_sus[uids]
    except Exception:
        return p_base       
    