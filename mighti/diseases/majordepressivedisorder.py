"""
Module defining Major Depressive Disorder as a remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease
from starsim.interventions import treat_num
import starsim as ss
import numpy as np


class MajorDepressiveDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'MajorDepressiveDisorder'
        super().__init__(csv_path, pars, **kwargs)

        # Add custom hospitalization-related parameters here
        self.define_pars(
            p_hospitalization_affected=0.01,
            p_hospitalization_treated=0.005,
            p_discharge_to_treatment=0.6,
            p_discharge_to_reversed=0.4,
            p_daily_discharge=0.1,
            p_daily_discharge_multiplier=1.0,
            rel_death_hospitalized=1.0,
            label='MajorDepressiveDisorder'
        )

        # Only set this if not already done via init
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 0.000203497644387431
    
    def define_states(self, *args, **kwargs):
        # Add custom states
        extra_states = [
            ss.State('hospitalized'),
            ss.FloatArr('ti_hospitalized', default=np.nan),
        ]
        # Call the base class's define_states with the combined states
        all_states = list(args) + extra_states
        super().define_states(*all_states, **kwargs)
        
    def step_state(self):
        super().step_state()
    
        # Hospitalization from affected
        affected_uids = self.affected.uids
        hosp_draws1 = np.random.rand(len(affected_uids))
        hosp_affected = affected_uids[hosp_draws1 < self.pars.p_hospitalization_affected]
        self.affected[hosp_affected] = False
        self.hospitalized[hosp_affected] = True
        self.ti_hospitalized[hosp_affected] = self.ti
        self.rel_death[hosp_affected] *= self.pars.rel_death_hospitalized
    
        # Hospitalization from on_treatment
        treated_uids = self.on_treatment.uids
        hosp_draws2 = np.random.rand(len(treated_uids))
        hosp_treated = treated_uids[hosp_draws2 < self.pars.p_hospitalization_treated]
        self.on_treatment[hosp_treated] = False
        self.hospitalized[hosp_treated] = True
        self.ti_hospitalized[hosp_treated] = self.ti
        self.rel_death[hosp_treated] *= self.pars.rel_death_hospitalized        
    
    def step_discharge(self):
        hospitalized_uids = self.hospitalized.uids
        if len(hospitalized_uids) == 0:
            return
    
        effective_p_discharge = self.pars.p_daily_discharge * self.pars.p_daily_discharge_multiplier
        draws = np.random.rand(len(hospitalized_uids))
        to_discharge = hospitalized_uids[draws < effective_p_discharge]
    
        route_draws = np.random.rand(len(to_discharge))
        to_treatment = to_discharge[route_draws < self.pars.p_discharge_to_treatment]
        to_reversed = np.setdiff1d(to_discharge, to_treatment)
    
        self.hospitalized[to_treatment] = False
        self.on_treatment[to_treatment] = True
    
        self.hospitalized[to_reversed] = False
        self.reversed[to_reversed] = True
        self.ti_reversed[to_reversed] = self.ti
    
        self.results.discharged[self.ti] = len(to_discharge)    

    def step(self):
        new_cases = super().step()
        self.step_discharge()
        return new_cases
    
    def init_results(self):
        super().init_results()
        if 'hospitalized_prevalence' not in self.results:
            self.define_results(ss.Result('hospitalized_prevalence', dtype=float, label='Hospitalized Prevalence'))
        if 'discharged' not in self.results:
            self.define_results(ss.Result('discharged', dtype=int, label='Hospital Discharges'))
    
    def update_results(self):
        super().update_results()
        self.results.hospitalized_prevalence[self.ti] = np.count_nonzero(self.hospitalized) / len(self.sim.people)


class DepressionCare(treat_num):
    def __init__(self, product=None, prob=1.0, max_capacity=None, eligibility=None, label='DepressionCare', **kwargs):
        super().__init__(product=product, prob=prob, max_capacity=max_capacity, eligibility=eligibility, label=label, **kwargs)
        self.disease = 'depression'  # used for default eligibility

    def initialize(self, sim):
        super().initialize(sim)
        # If no custom eligibility is given, treat all affected
        if self.eligibility is None:
            if not hasattr(sim.diseases, self.disease):
                raise ValueError(f"[{self.label}] Disease '{self.disease}' not found in sim.diseases.")
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids

    def step(self):
        self.add_to_queue()
        treat_inds = super().step()
        if len(treat_inds):
            print(f"[{self.label}] Treated {len(treat_inds)} agents for depression at step {self.ti}")
        return treat_inds
    
