"""
Defines health conditions and their base logic, including disease-specific behavior and initialization.
"""

import logging
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import lognorm


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
    }


class RemittingDisease(ss.NCD):
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
            p_acquire=disease_params["p_acquire"],
            init_prev=None
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
            affected = np.random.rand(len(self.sim.people)) < probs       # ← fixed

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
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(self.sim.people, 'hiv'):
                hiv_pos = self.sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        draws = np.random.rand(len(susceptible))
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
        draws = np.random.rand(len(affected_uids))
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


class AcuteDisease(ss.NCD):
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
            p_acquire=disease_params["p_acquire"],
            init_prev=None,
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
            affected = np.random.rand(len(sim.people)) < probs

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
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(self.sim.people, 'hiv'):
                hiv_pos = self.sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]

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
    


class AcuteSurgicalDisease(ss.NCD):
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
            p_acquire=disease_params["p_acquire"],
            p_surgery=disease_params.get("p_surgery", 0.3),
            rel_mortality_treated=disease_params.get("rel_mortality_treated", 0.5),
            rel_mortality_untreated=disease_params.get("rel_mortality_untreated", 2.0),
            cost_surgery=disease_params.get("cost_surgery", 0.0),
            init_prev=None,
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
            affected = np.random.rand(len(sim.people)) < probs
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

        # --- Acquisition ---
        susceptible = self.at_risk.uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(sim.people, "hiv"):
                hiv_pos = sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti
        self.rel_death[new_cases] = self.pars.rel_mortality_untreated

        # --- Surgery events ---
        affected_uids = self.affected.uids
        can_surgery = affected_uids[~self.surgery_done[affected_uids]]
        surgeries = can_surgery[np.random.rand(len(can_surgery)) < self.pars.p_surgery]
        if len(surgeries):
            self.on_treatment[surgeries] = True
            self.surgery_done[surgeries] = True
            self.ti_surgery[surgeries] = ti
            self.rel_death[surgeries] = self.pars.rel_mortality_treated

        # --- Deaths ---
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get("p", 0)
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]
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
    

class ChronicDisease(ss.NCD):
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
            p_acquire=disease_params["p_acquire"],
            init_prev=None,
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
            affected = np.random.rand(len(sim.people)) < probs

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
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(self.sim.people, 'hiv'):
                hiv_pos = self.sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]

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
    

class GenericSIS(ss.SIS):
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
            p_acquire=disease_params["p_acquire"],
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
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            reset=True,
        )

    def init_post(self):
        super().init_post()

        sim = self.sim  # Starsim assigns this automatically in init_pre(sim)

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            # Sample prevalence probabilities
            probs = self.pars.init_prev.rvs(sim.people.uid)
            infected = np.random.rand(len(sim.people)) < probs

            # Assign disease state
            if hasattr(self, "infected"):
                self.infected[:] = infected 

            # Optionally set prognoses for infected agents
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(infected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.infected[uids] = True
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
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        susceptible = self.at_risk.uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(self.sim.people, 'hiv'):
                hiv_pos = self.sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        self.infected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_infected[new_cases] = ti

        # Deaths
        affected_uids = self.infected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]

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


class GenericSIR(ss.SIR):
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
            p_acquire=disease_params["p_acquire"],             # force of infection term
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
        super().init_post()
        sim = self.sim

        # Initialize infection prevalence if provided
        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            probs = self.pars.init_prev.rvs(sim.people.uid)
            init_inf = np.random.rand(len(sim.people)) < probs

            self.infected[init_inf]   = True
            self.susceptible[init_inf] = False
            self.at_risk[init_inf]     = False
            self.ti_infected[init_inf] = self.ti

    def set_prognoses(self, uids):
        # Enter I from S
        self.susceptible[uids] = False
        self.infected[uids]    = True
        self.at_risk[uids]     = False

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

        # --- Acquire infection (S → I) ---
        susceptible = self.at_risk.uids & self.susceptible.uids  # ensure truly in S
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        # Sex filtering
        if self.pars.affected_sex == "female":
            p_acq[sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[sim.people.female[susceptible]] = 0

        # Modifiers
        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(sim.people, 'hiv'):
                hiv_pos = sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
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
        deaths = infected_uids[np.random.rand(len(infected_uids)) < base_p * (rel_death if len(rel_death) else 1.0)]
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
    
    
class NonAcquiredDisease(ss.Module):
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

        # Draw initial affected status
        if hasattr(self.pars.init_prev, "rvs"):
            affected = self.pars.init_prev.rvs(sim.people.uid)
        elif callable(self.pars.init_prev):
            affected = np.array(self.pars.init_prev(), dtype=bool)
        else:
            p = float(self.pars.init_prev)
            affected = np.random.rand(n) < p

        self.affected[:] = affected
        self.ti_affected[affected] = self.ti

        n_affected = affected.sum()
        logger.info(f"[INIT] {self.disease_name}: {n_affected}/{n} ({n_affected/n:.3%}) affected at birth")

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
        # Skip the very first timestep so neonatal deaths are not applied before survivorship baseline
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
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]

        if len(deaths):
            sim.people.request_death(deaths)
            logger.debug(f"[STEP] {self.disease_name}: {len(deaths)} deaths at timestep {ti}")

        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        self.results.n_affected[ti] = np.count_nonzero(self.affected)

    def step_die(self, uids):
        """Record death times for affected individuals."""
        if len(uids):
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
        affected = self.affected.uids
        if not len(affected):
            return np.array([])

        rel_death = self.rel_death[affected]
        base_p = self.pars.p_death.pars.get("p", 0)
        deaths = affected[np.random.rand(len(affected)) < base_p * rel_death]
        if len(deaths):
            sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        return deaths    

def calculate_p_acquire_generic(disease, sim, uids):
    """Calculate acquisition probability for a disease with optional sex filtering and HIV interaction."""
    p_base = np.full(len(uids), disease.pars.p_acquire_multiplier * disease.pars.p_acquire)
    
    if disease.pars.affected_sex == "female":
        try:
            p_base[sim.people.male[uids]] = 0
        except Exception:
            pass
    elif disease.pars.affected_sex == "male":
        try:
            p_base[sim.people.female[uids]] = 0
        except Exception:
            pass

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
    