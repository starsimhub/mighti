"""
Defines health conditions and their base logic, including disease-specific behavior and initialization.
"""

import logging
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import lognorm


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
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.State('on_treatment'),
            ss.State('reversed'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),  
        )

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

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

        # New cases
        susceptible = (~self.affected).uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        # Apply sex filtering
        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        # Apply rel_sus and rel_sus_hiv
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
            init_prev=None
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.State('on_treatment'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),  
        )

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
    
        self.susceptible[uids] = False
        self.affected[uids] = True
    
        dur_condition = p.dur_condition.rvs(size=len(uids))
    
        dead_uids = p.p_death.filter(uids)    
        rec_uids = np.setdiff1d(uids, dead_uids)
        dead_indices = np.isin(uids, dead_uids)
        rec_indices = ~dead_indices
    
        self.ti_dead[dead_uids] = self.ti + dur_condition[dead_indices] / self.t.dt
    
        if hasattr(self, "ti_reversed"):
            self.ti_reversed[rec_uids] = self.ti + dur_condition[rec_indices] / self.t.dt
    
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

        # New cases
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
        
        # Sample new cases using numpy
        draws = np.random.rand(len(susceptible))
        new_cases = susceptible[draws < p_acq]
        self.affected[new_cases] = True
        self.ti_affected[new_cases] = ti
                
        # New implementation of detah
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]

        try:
            base_p = self.pars.p_death.pars['p']
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
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.State('on_treatment'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),  
        )
    
    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
    
        self.susceptible[uids] = False
        self.affected[uids] = True
    
        dur_condition = p.dur_condition.rvs(size=len(uids))
    
        dead_uids = p.p_death.filter(uids)    
        rec_uids = np.setdiff1d(uids, dead_uids)
        dead_indices = np.isin(uids, dead_uids)
        rec_indices = ~dead_indices
    
        self.ti_dead[dead_uids] = self.ti + dur_condition[dead_indices] / self.t.dt
    
        if hasattr(self, "ti_reversed"):
            self.ti_reversed[rec_uids] = self.ti + dur_condition[rec_indices] / self.t.dt

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

        # New cases
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
        
        # Sample new cases using numpy
        draws = np.random.rand(len(susceptible))
        new_cases = susceptible[draws < p_acq]
        self.affected[new_cases] = True
        self.ti_affected[new_cases] = ti
                
        # New implementation of detah
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]

        try:
            base_p = self.pars.p_death.pars['p']
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
            init_prev=None
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate) 

        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('infected'),
            ss.State('on_treatment'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0), 
            ss.FloatArr('rel_death', default=1.0), 
        )

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
    
        self.susceptible[uids] = False
        self.infected[uids] = True
    
        dur_condition = p.dur_condition.rvs(size=len(uids))
    
        dead_uids = p.p_death.filter(uids)    
        rec_uids = np.setdiff1d(uids, dead_uids)
        dead_indices = np.isin(uids, dead_uids)
        rec_indices = ~dead_indices
    
        self.ti_dead[dead_uids] = self.ti + dur_condition[dead_indices] / self.t.dt
    
        if hasattr(self, "ti_reversed"):
            self.ti_reversed[rec_uids] = self.ti + dur_condition[rec_indices] / self.t.dt

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

        # New cases
        susceptible = (~self.infected).uids
        
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
        
        # Sample new cases using numpy
        draws = np.random.rand(len(susceptible))
        new_cases = susceptible[draws < p_acq]

        self.infected[new_cases] = True
        self.ti_infected[new_cases] = ti
                
        # New implementation of detah
        affected_uids = self.infected.uids
        rel_death = self.rel_death[affected_uids]

        try:
            base_p = self.pars.p_death.pars['p']
        except Exception:
            raise ValueError(f"Cannot extract base death probability from {self.pars.p_death}")

        adjusted_p_death = base_p * rel_death
        draws = np.random.rand(len(affected_uids))
        deaths = affected_uids[draws < adjusted_p_death]

        self.sim.people.request_death(deaths)

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

        return new_cases   
    

def calculate_p_acquire_generic(disease, sim, uids):
    """Simplified for calibration: assume constant baseline, scaled by multiplier."""

    # Optional: limit to adults
    age = sim.people.age[uids]
    p_base = np.zeros(len(uids))
    adult = age >= 15
    p_base[adult] = disease.pars.p_acquire_multiplier

    # Optional: restrict by sex
    if disease.pars.affected_sex == "female":
        p_base[sim.people.male[uids]] = 0
    elif disease.pars.affected_sex == "male":
        p_base[sim.people.female[uids]] = 0

    return p_base



class Type1Diabetes(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type1Diabetes'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label='Type1Diabetes')  

        return


class Type2Diabetes(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        print("[CALIB] Initializing Type2Diabetes for calibration")
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


class TobaccoUse(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'TobaccoUse'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'TobaccoUse')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return

class HIVAssociatedDementia(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'HIVAssociatedDementia'
        super().__init__(csv_path, pars, **kwargs)
       
        self.define_pars(label = 'HIVAssociatedDementia')
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


class Depression(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Depression'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Depression')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return


class DomesticViolence(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DomesticViolence'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'DomesticViolence')
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


class ViralHepatitis(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ViralHepatitis'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ViralHepatitis')
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


class TB(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'TB'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'TB')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire_multiplier = 1  
        return
    
