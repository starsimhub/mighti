"""
Defines health conditions and their base logic, including disease-specific behavior and initialization.
"""

import logging
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import lognorm

from mighti.util.rng import get_rng


__all__ = ['RemittingDisease', 'AcuteDisease', 'ChronicDisease', 'GenericSIS']


logger = logging.getLogger(__name__)


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
    }


class RemittingDisease(ss.NCD):
    """ Base class for all remitting diseases."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path        
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)        

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
            p_acquire=1,
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

        sim = self.sim  # Starsim assigns this automatically in init_pre(sim)

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

            # Handle recovery, beta-cell function exhaustion
            recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
            self.reversed[recovered] = False
            self.susceptible[recovered] = True  

    def step(self):
        ti = self.ti
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")

        # New cases
        susceptible = (~self.affected).uids
        # p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)
        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

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


class AcuteDisease(ss.NCD):
    """ Base class for all acute diseases. """

    def __init__(self, csv_path=None, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
                
        # Calculate the mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        # Define parameters using extracted values
        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
            p_death=ss.bernoulli(disease_params["p_death"]),  
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  
            affected_sex=disease_params["affected_sex"],
            p_acquire=1,
            # Starsim base classes often define `init_prev` as a distribution.
            # Setting it to None can trigger "Updating dist ... to NoneType not supported".
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

        sim = self.sim  # Starsim assigns this automatically in init_pre(sim)

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

    def set_prognoses(self, uids, sources=None, **kwargs):  # noqa: ARG002
        sim = self.sim
        p = self.pars
    
        self.susceptible[uids] = False
        self.affected[uids] = True
    
        dur_condition = p.dur_condition.rvs(size=len(uids))
    
        dead_uids = p.p_death.filter(uids)    
        rec_uids = np.setdiff1d(uids, dead_uids)
        dead_indices = np.isin(uids, dead_uids)
        rec_indices = ~dead_indices

        # Convert durations (years) into numeric timesteps and add to current ti.
        # Avoid Starsim time/rate objects here; ti_* arrays expect numeric timestep indices.
        try:
            ti0 = float(sim.ti)
        except Exception:
            ti0 = float(self.ti)
        try:
            dt = float(getattr(sim, "dt", getattr(self.t, "dt", 1.0)))
        except Exception:
            dt = 1.0
        dt = dt if dt > 0 else 1.0

        dur_condition = np.asarray(dur_condition, dtype=float)
        offset = np.maximum(dur_condition / dt, 0.0)

        self.ti_dead[dead_uids] = ti0 + offset[dead_indices]

        if hasattr(self, "ti_reversed"):
            self.ti_reversed[rec_uids] = ti0 + offset[rec_indices]
    
    def init_results(self):
        super().init_results()
        existing_results = set(self.results.keys())
        
        if 'new_cases' not in existing_results:
            self.define_results(ss.Result('new_cases', dtype=int, label='New Cases'))
        if 'new_deaths' not in existing_results:
            self.define_results(ss.Result('new_deaths', dtype=int, label='Deaths'))
        if 'prevalence' not in existing_results:
            self.define_results(ss.Result('prevalence', dtype=float, label='Prevalence'))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        susceptible = self.at_risk.uids
        # p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)
        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

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

        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[rng.random(len(affected_uids)) < base_p * rel_death]

        self.sim.people.request_death(deaths)
        self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return new_cases


class ChronicDisease(ss.NCD):
    """ Base class for all chronic diseases. """

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        
        # Calculate the mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2
    
        # Define parameters using extracted values
        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
            p_death=ss.bernoulli(disease_params["p_death"]),  
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  
            affected_sex=disease_params["affected_sex"],
            p_acquire=1,
            init_prev=None
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
        sim = self.sim  # Starsim assigns this automatically in init_pre(sim)

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
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
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

        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[rng.random(len(affected_uids)) < base_p * rel_death]

        self.sim.people.request_death(deaths)
        self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return new_cases


class GenericSIS(ss.SIS):
    """ Base class for communicable diseases using the SIS model. """

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)        
        
        # Define parameters using extracted values
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
            p_acquire=1,
            # See AcuteDisease: avoid changing dist types by setting None.
            init_prev=ss.bernoulli(0.0),
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
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            reset=True,
        )

    def init_post(self):
        # Use Starsim Infection.init_post(), which seeds infections via
        # `init_prev.filter()` and calls `set_prognoses(initial_cases, sources=-1)`.
        return super().init_post()


    def set_prognoses(self, uids, sources=None, **kwargs):  # noqa: ARG002
        sim = self.sim
        p = self.pars
    
        self.susceptible[uids] = False
        self.infected[uids] = True

        try:
            ti0 = float(sim.ti)
        except Exception:
            ti0 = float(self.ti)
        self.ti_infected[uids] = ti0
    
        dur_condition = p.dur_condition.rvs(size=len(uids))
    
        dead_uids = p.p_death.filter(uids)    
        rec_uids = np.setdiff1d(uids, dead_uids)
        dead_indices = np.isin(uids, dead_uids)
        rec_indices = ~dead_indices

        # Numeric timestep arithmetic only (avoid Starsim time/rate objects here)
        try:
            dt = float(getattr(sim, "dt", getattr(self.t, "dt", 1.0)))
        except Exception:
            dt = 1.0
        dt = dt if dt > 0 else 1.0

        dur_condition = np.asarray(dur_condition, dtype=float)
        offset = np.maximum(dur_condition / dt, 0.0)

        self.ti_dead[dead_uids] = ti0 + offset[dead_indices]

        self.ti_recovered[rec_uids] = ti0 + offset[rec_indices]

    def step(self):
        """Acquire new infections using `calculate_p_acquire_generic()`."""
        ti = self.ti
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")

        susceptible = self.susceptible.uids
        if len(susceptible) == 0:
            return np.array([], dtype=int)

        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq = p_acq * self.rel_sus[susceptible]
        except Exception:
            pass

        p_acq = np.clip(p_acq, 0.0, 1.0)
        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        if len(new_cases):
            self.set_prognoses(new_cases, sources=-1)
        return new_cases

    def step_state(self):
        """Progress infected -> susceptible when `ti_recovered` is reached."""
        ti = self.ti
        recovered = (self.infected & (self.ti_recovered <= ti)).uids
        if len(recovered):
            self.infected[recovered] = False
            self.susceptible[recovered] = True
        return


class GenericSIR(ss.SIR):
    """Base class for communicable diseases using an SIR-style model."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)

        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # duration in years (we convert to timesteps)
            p_death=ss.bernoulli(disease_params["p_death"]),
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire=1,
            init_prev=ss.bernoulli(0.0),
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState("susceptible", default=True),
            ss.BoolState("at_risk", default=True),
            ss.BoolState("infected"),
            ss.BoolState("recovered"),
            ss.BoolState("on_treatment"),
            ss.FloatArr("ti_infected"),
            ss.FloatArr("ti_recovered"),
            ss.FloatArr("ti_dead"),
            ss.FloatArr("rel_sus", default=1.0),
            ss.FloatArr("rel_death", default=1.0),
            reset=True,
        )

    def init_post(self):
        return super().init_post()

    def set_prognoses(self, uids, sources=None, **kwargs):  # noqa: ARG002
        """Set prognoses for new infections, using numeric timestep arithmetic."""
        sim = self.sim
        p = self.pars

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.recovered[uids] = False

        ti = float(sim.ti)
        self.ti_infected[uids] = ti

        dur_condition = np.asarray(p.dur_condition.rvs(size=len(uids)), dtype=float)

        dead_uids = p.p_death.filter(uids)
        rec_uids = np.setdiff1d(uids, dead_uids)
        dead_indices = np.isin(uids, dead_uids)
        rec_indices = ~dead_indices

        try:
            dt = float(getattr(sim, "dt", getattr(self.t, "dt", 1.0)))
        except Exception:
            dt = 1.0
        dt = dt if dt > 0 else 1.0

        offset = np.maximum(dur_condition / dt, 0.0)
        self.ti_dead[dead_uids] = ti + offset[dead_indices]
        self.ti_recovered[rec_uids] = ti + offset[rec_indices]
        return

    def step(self):
        """Acquire new infections using `calculate_p_acquire_generic()`."""
        ti = self.ti
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")

        susceptible = self.susceptible.uids
        if len(susceptible) == 0:
            return np.array([], dtype=int)

        p_acq = calculate_p_acquire_generic(self, self.sim, susceptible)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq = p_acq * self.rel_sus[susceptible]
        except Exception:
            pass

        p_acq = np.clip(p_acq, 0.0, 1.0)
        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        if len(new_cases):
            self.set_prognoses(new_cases, sources=-1)
        return new_cases

    def init_results(self):
        super().init_results()
        existing_results = set(self.results.keys())
        
        if 'new_cases' not in existing_results:
            self.define_results(ss.Result('new_cases', dtype=int, label='New Cases'))
        if 'new_deaths' not in existing_results:
            self.define_results(ss.Result('new_deaths', dtype=int, label='Deaths'))
        if 'prevalence' not in existing_results:
            self.define_results(ss.Result('prevalence', dtype=float, label='Prevalence'))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
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

        new_cases = susceptible[rng.random(len(susceptible)) < p_acq]
        self.infected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_infected[new_cases] = ti

        # Deaths
        affected_uids = self.infected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[rng.random(len(affected_uids)) < base_p * rel_death]

        self.sim.people.request_death(deaths)
        self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)
        return new_cases
    

def calculate_p_acquire_generic(disease, sim, uids):
    """Calculate acquisition probability for a disease with optional sex filtering and HIV interaction."""
    p_base = np.full(len(uids), disease.pars.p_acquire_multiplier)
    
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
       



class Type1Diabetes(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type1Diabetes'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label='Type1Diabetes')  

        return


class Type2Diabetes(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type2Diabetes'
        super().__init__(csv_path, pars, **kwargs)

        self.define_pars(label='Type2Diabetes')  

        return
    

class Hypertension(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Hypertension'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Hypertension')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class Obesity(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Obesity'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Obesity')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class CardiovascularDiseases(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'CardiovascularDiseases'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'CardiovascularDiseases')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class ChronicKidneyDisease(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ChronicKidneyDisease'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ChronicKidneyDisease')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class Hyperlipidemia(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Hyperlipidemia'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Hyperlipidemia')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    

class CervicalCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'CervicalCancer'
        super().__init__(csv_path, pars, **kwargs)
       
        self.define_pars(label = 'CervicalCancer')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class ColorectalCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ColorectalCancer'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ColorectalCancer')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class BreastCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'BreastCancer'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'BreastCancer')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    

class LungCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'LungCancer'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'LungCancer')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    

class ProstateCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ProstateCancer'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ProstateCancer')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class AlcoholUseDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AlcoholUseDisorder'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'AlcoholUseDisorder')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class AnxietyDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AnxietyDisorder'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'AnxietyDisorder')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    
    
class ChronicPain(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ChronicPain'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ChronicPain')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    
    
class DrugUseDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DrugUseDisorder'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'DrugUseDisorder')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return    
    

class OpioidUseDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'OpioidUseDisorder'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'OpioidUseDisorder')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    
    
class TobaccoUse(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'TobaccoUse'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'TobaccoUse')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    

class Dementia(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Dementia'
        super().__init__(csv_path, pars, **kwargs)
       
        self.define_pars(label = 'Dementia')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class PTSD(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'PTSD'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'PTSD')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class MajorDepressiveDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'MajorDepressiveDisorder'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'MajorDepressiveDisorder')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
        

class RoadInjuries(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'RoadInjuries'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'RoadInjuries')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class ChronicLiverDisease(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ChronicLiverDisease'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ChronicLiverDisease')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class Asthma(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Asthma'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Asthma')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class COPD(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'COPD'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'COPD')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class AlzheimersDisease(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AlzheimersDisease'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'AlzheimersDisease')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class ParkinsonsDisease(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ParkinsonsDisease'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ParkinsonsDisease')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class AcuteHepatitis(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AcuteHepatitis'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'AcuteHepatitis')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class HPV(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'HPV'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'HPV')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class Flu(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Flu'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Flu')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class Tuberculosis(GenericSIR):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Tuberculosis'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Tuberculosis')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class SelfHarm(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'SelfHarm'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'SelfHarm')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class MaternalConditions(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'MaternalConditions'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'MaternalConditions')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class DiarrhealDiseases(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DiarrhealDiseases'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'DiarrhealDiseases')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class LowerRespiratoryInfections(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'LowerRespiratoryInfections'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'LowerRespiratoryInfections')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return

class COVID19(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'COVID19'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'COVID19')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return

class InterpersonalViolence(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'InterpersonalViolence'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'InterpersonalViolence')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
