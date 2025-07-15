"""
Defines health conditions and their base logic, including disease-specific behavior and initialization.
"""

import logging
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import lognorm


__all__ = ['AlcoholUseDisorder', 'SubstanceUseDisorder', 'Depression']


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


class AlcoholUseDisorder(ss.NCD):

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.disease_name = 'AlcoholUseDisorder'
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
            p_acquire_multiplier=1.0,
            init_prev=None,
            p_hospitalization_affected=0.01,  # daily prob from affected → hospitalized
            p_hospitalization_treated=0.005,  # daily prob from on_treatment → hospitalized
            p_discharge_to_treatment=0.6,     # prob that discharge goes to on_treatment
            p_discharge_to_reversed=0.4,      # prob that discharge goes to reversed
            p_daily_discharge=0.1,            # prob of discharging per day
            p_daily_discharge_multiplier = 1.0,  # default, can be increased by intervention
            rel_death_hospitalized=1.0,
            label='AlcoholUseDisorder'
        )
        
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 0.018  # After calibration
        
        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate) 

        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.State('on_treatment'),
            ss.State('reversed'), 
            ss.State('hospitalized'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),  
        )

    def init_post(self):
        if 'init_prev' in self.pars:
            initial_cases = self.pars.init_prev.filter()
            self.set_prognoses(initial_cases)
            return initial_cases
        return []

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
        if 'hospitalized_prevalence' not in existing_results:
            self.define_results(ss.Result('hospitalized_prevalence', dtype=float, label='Hospitalization Prevalence'))
        if 'discharged' not in existing_results:
            self.define_results(ss.Result('discharged', dtype=int, label='Hospital Discharges'))       

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        self.results.hospitalized_prevalence[self.ti] = np.count_nonzero(self.hospitalized) / len(self.sim.people)

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
            
        # Hospitalization
        # From affected
        affected_uids = self.affected.uids
        hosp_draws1 = np.random.rand(len(affected_uids))
        hosp_affected = affected_uids[hosp_draws1 < self.pars.p_hospitalization_affected]
        self.affected[hosp_affected] = False
        self.hospitalized[hosp_affected] = True
    
        # From on_treatment
        treated_uids = self.on_treatment.uids
        hosp_draws2 = np.random.rand(len(treated_uids))
        hosp_treated = treated_uids[hosp_draws2 < self.pars.p_hospitalization_treated]
        self.on_treatment[hosp_treated] = False
        self.hospitalized[hosp_treated] = True
        
        # Adjust rel_death for newly hospitalized
        self.rel_death[hosp_affected] *= self.pars.rel_death_hospitalized
        self.rel_death[hosp_treated] *= self.pars.rel_death_hospitalized

    def step_discharge(self):
        hospitalized_uids = self.hospitalized.uids
        if len(hospitalized_uids) == 0:
            return
    
        # Daily discharge
        effective_p_discharge = self.pars.p_daily_discharge * self.pars.p_daily_discharge_multiplier
        draws = np.random.rand(len(hospitalized_uids))  # ← you forgot this line
        to_discharge = hospitalized_uids[draws < effective_p_discharge]
    
        # Split between treatment and reversal
        route_draws = np.random.rand(len(to_discharge))
        to_treatment = to_discharge[route_draws < self.pars.p_discharge_to_treatment]
        to_reversed = np.setdiff1d(to_discharge, to_treatment)
    
        # Transition
        self.hospitalized[to_treatment] = False
        self.on_treatment[to_treatment] = True
    
        self.hospitalized[to_reversed] = False
        self.reversed[to_reversed] = True
        self.ti_reversed[to_reversed] = self.ti
    
        self.results.discharged[self.ti] = len(to_discharge)
        
    def step(self):
        ti = self.ti
        
        # Step internal transitions first
        self.step_state()
        self.step_discharge()


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


class SubstanceUseDisorder(ss.NCD):

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.disease_name = 'SubstanceUseDisorder'
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
            p_acquire_multiplier=1.0,
            init_prev=None,
            p_hospitalization_affected=0.01,  # daily prob from affected → hospitalized
            p_hospitalization_treated=0.005,  # daily prob from on_treatment → hospitalized
            p_discharge_to_treatment=0.6,     # prob that discharge goes to on_treatment
            p_discharge_to_reversed=0.4,      # prob that discharge goes to reversed
            p_daily_discharge=0.1,            # prob of discharging per day
            p_daily_discharge_multiplier = 1.0,  # default, can be increased by intervention
            rel_death_hospitalized=1.0,
            label='SubstanceUseDisorder'
        )
        
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 0.018  # After calibration
        
        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate) 

        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.State('on_treatment'),
            ss.State('reversed'), 
            ss.State('hospitalized'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),  
        )

    def init_post(self):
        if 'init_prev' in self.pars:
            initial_cases = self.pars.init_prev.filter()
            self.set_prognoses(initial_cases)
            return initial_cases
        return []

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
        if 'hospitalized_prevalence' not in existing_results:
            self.define_results(ss.Result('hospitalized_prevalence', dtype=float, label='Hospitalization Prevalence'))
        if 'discharged' not in existing_results:
            self.define_results(ss.Result('discharged', dtype=int, label='Hospital Discharges'))       

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        self.results.hospitalized_prevalence[self.ti] = np.count_nonzero(self.hospitalized) / len(self.sim.people)

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
            
        # Hospitalization
        # From affected
        affected_uids = self.affected.uids
        hosp_draws1 = np.random.rand(len(affected_uids))
        hosp_affected = affected_uids[hosp_draws1 < self.pars.p_hospitalization_affected]
        self.affected[hosp_affected] = False
        self.hospitalized[hosp_affected] = True
    
        # From on_treatment
        treated_uids = self.on_treatment.uids
        hosp_draws2 = np.random.rand(len(treated_uids))
        hosp_treated = treated_uids[hosp_draws2 < self.pars.p_hospitalization_treated]
        self.on_treatment[hosp_treated] = False
        self.hospitalized[hosp_treated] = True
        
        # Adjust rel_death for newly hospitalized
        self.rel_death[hosp_affected] *= self.pars.rel_death_hospitalized
        self.rel_death[hosp_treated] *= self.pars.rel_death_hospitalized

    def step_discharge(self):
        hospitalized_uids = self.hospitalized.uids
        if len(hospitalized_uids) == 0:
            return
    
        # Daily discharge
        effective_p_discharge = self.pars.p_daily_discharge * self.pars.p_daily_discharge_multiplier
        draws = np.random.rand(len(hospitalized_uids))  # ← you forgot this line
        to_discharge = hospitalized_uids[draws < effective_p_discharge]
    
        # Split between treatment and reversal
        route_draws = np.random.rand(len(to_discharge))
        to_treatment = to_discharge[route_draws < self.pars.p_discharge_to_treatment]
        to_reversed = np.setdiff1d(to_discharge, to_treatment)
    
        # Transition
        self.hospitalized[to_treatment] = False
        self.on_treatment[to_treatment] = True
    
        self.hospitalized[to_reversed] = False
        self.reversed[to_reversed] = True
        self.ti_reversed[to_reversed] = self.ti
    
        self.results.discharged[self.ti] = len(to_discharge)
        
    def step(self):
        ti = self.ti
        
        # Step internal transitions first
        self.step_state()
        self.step_discharge()


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


class Depression(ss.NCD):

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.disease_name = 'Depression'
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
            p_acquire_multiplier=1.0,
            init_prev=None,
            p_hospitalization_affected=0.01,  # daily prob from affected → hospitalized
            p_hospitalization_treated=0.005,  # daily prob from on_treatment → hospitalized
            p_discharge_to_treatment=0.6,     # prob that discharge goes to on_treatment
            p_discharge_to_reversed=0.4,      # prob that discharge goes to reversed
            p_daily_discharge=0.1,            # prob of discharging per day
            p_daily_discharge_multiplier = 1.0,  # default, can be increased by intervention
            rel_death_hospitalized=1.0,
            label='Depression'
        )
        
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 0.018  # After calibration
        
        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate) 

        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.State('on_treatment'),
            ss.State('reversed'), 
            ss.State('hospitalized'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),  
        )

    def init_post(self):
        if 'init_prev' in self.pars:
            initial_cases = self.pars.init_prev.filter()
            self.set_prognoses(initial_cases)
            return initial_cases
        return []

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
        if 'hospitalized_prevalence' not in existing_results:
            self.define_results(ss.Result('hospitalized_prevalence', dtype=float, label='Hospitalization Prevalence'))
        if 'discharged' not in existing_results:
            self.define_results(ss.Result('discharged', dtype=int, label='Hospital Discharges'))       

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        self.results.hospitalized_prevalence[self.ti] = np.count_nonzero(self.hospitalized) / len(self.sim.people)

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
            
        # Hospitalization
        # From affected
        affected_uids = self.affected.uids
        hosp_draws1 = np.random.rand(len(affected_uids))
        hosp_affected = affected_uids[hosp_draws1 < self.pars.p_hospitalization_affected]
        self.affected[hosp_affected] = False
        self.hospitalized[hosp_affected] = True
    
        # From on_treatment
        treated_uids = self.on_treatment.uids
        hosp_draws2 = np.random.rand(len(treated_uids))
        hosp_treated = treated_uids[hosp_draws2 < self.pars.p_hospitalization_treated]
        self.on_treatment[hosp_treated] = False
        self.hospitalized[hosp_treated] = True
        
        # Adjust rel_death for newly hospitalized
        self.rel_death[hosp_affected] *= self.pars.rel_death_hospitalized
        self.rel_death[hosp_treated] *= self.pars.rel_death_hospitalized

    def step_discharge(self):
        hospitalized_uids = self.hospitalized.uids
        if len(hospitalized_uids) == 0:
            return
    
        # Daily discharge
        effective_p_discharge = self.pars.p_daily_discharge * self.pars.p_daily_discharge_multiplier
        draws = np.random.rand(len(hospitalized_uids))  # ← you forgot this line
        to_discharge = hospitalized_uids[draws < effective_p_discharge]
    
        # Split between treatment and reversal
        route_draws = np.random.rand(len(to_discharge))
        to_treatment = to_discharge[route_draws < self.pars.p_discharge_to_treatment]
        to_reversed = np.setdiff1d(to_discharge, to_treatment)
    
        # Transition
        self.hospitalized[to_treatment] = False
        self.on_treatment[to_treatment] = True
    
        self.hospitalized[to_reversed] = False
        self.reversed[to_reversed] = True
        self.ti_reversed[to_reversed] = self.ti
    
        self.results.discharged[self.ti] = len(to_discharge)
        
    def step(self):
        ti = self.ti
        
        # Step internal transitions first
        self.step_state()
        self.step_discharge()


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
    