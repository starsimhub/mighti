import numpy as np
from scipy.stats import bernoulli, lognorm
import starsim as ss
import mighti as mi

# Define Type2Diabetes condition
class Type2Diabetes(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            dur_condition=lognorm(s=0.5, scale=np.exp(1.5)),  # Log-normal distribution for duration
            incidence_prob=0.0315,
            p_death=bernoulli(0.0017),  # Define p_death as a Bernoulli distribution
            init_prev=0.2,
            remission_rate=bernoulli(0.0024),  # Define remission_rate as a Bernoulli distribution
            max_disease_duration=20,
        )

        # Define disease parameters
        self.define_pars(
            p_acquire=0.0315,  # Probability of acquisition per timestep
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_acquire * self.rel_sus[uids])
        self.p_death = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_death.mean())  # Use mean to match Bernoulli
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate.mean())  # Use mean to match Bernoulli

        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.State('reversed'),  # New state for diabetes remission
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),  # Relative susceptibility
            ss.FloatArr('rel_death', default=1.0),  # Relative mortality
        )
        return

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
        will_die = p.p_death.rvs(size=len(uids))
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt
        self.ti_reversed[rec_uids] = self.ti + dur_condition[~will_die] / self.t.dt
        return

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
        if 'reversal_prevalence' not in existing_results:
            self.define_results(ss.Result('reversal_prevalence', dtype=float))
        
        return

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.reversal_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        return

    def step_state(self):
        # Handle remission (reversal)
        going_into_remission = self.p_remission.filter(self.affected.uids)  # Use p_remission for filtering
        self.affected[going_into_remission] = False
        self.reversed[going_into_remission] = True
        self.ti_reversed[going_into_remission] = self.ti

        # Handle recovery, death, and beta-cell function exhaustion
        recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
        self.reversed[recovered] = False
        self.susceptible[recovered] = True  # Recovered individuals become susceptible again
        deaths = (self.ti_dead == self.ti).uids
        self.sim.people.request_death(deaths)
        self.results.new_deaths[self.ti] = len(deaths)

    def step(self):
        ti = self.ti

        # New cases
        susceptible = (~self.affected).uids
        new_cases = self.p_acquire.filter(susceptible)
        self.affected[new_cases] = True
        self.ti_affected[new_cases] = ti

        # Death
        deaths = self.p_death.filter(new_cases)  # Applied only to people just affected
        self.sim.people.request_death(deaths)
        self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

        return new_cases

# Define Chronic Kidney Disease condition
class ChronicKidneyDisease(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            dur_condition=lognorm(s=0.5, scale=np.exp(1.5)),  # Log-normal distribution for duration
            incidence_prob=0.015,  # Lower incidence probability than T2D
            p_death=bernoulli(0.005),  # Higher probability of death than T2D
            init_prev=0.1,
            max_disease_duration=30,
        )

        # Define disease parameters
        self.define_pars(
            p_acquire=0.015,  # Probability of acquisition per timestep
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_acquire * self.rel_sus[uids])
        self.p_death = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_death.mean())  # Use mean to match Bernoulli

        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),  # Relative susceptibility
            ss.FloatArr('rel_death', default=1.0),  # Relative mortality
        )
        return

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
        will_die = p.p_death.rvs(size=len(uids))
        dead_uids = uids[will_die]
        self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt
        return

    def init_results(self):
        super().init_results()
        existing_results = set(self.results.keys())
        
        if 'new_cases' not in existing_results:
            self.define_results(ss.Result('new_cases', dtype=int, label='New Cases'))
        if 'new_deaths' not in existing_results:
            self.define_results(ss.Result('new_deaths', dtype=int, label='Deaths'))
        if 'prevalence' not in existing_results:
            self.define_results(ss.Result('prevalence', dtype=float, label='Prevalence'))
        
        return

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return

    def step(self):
        ti = self.ti

        # New cases
        susceptible = (~self.affected).uids
        new_cases = self.p_acquire.filter(susceptible)
        self.affected[new_cases] = True
        self.ti_affected[new_cases] = ti

        # Death
        deaths = self.p_death.filter(new_cases)  # Applied only to people just affected
        self.sim.people.request_death(deaths)
        self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(self.sim.people)

        return new_cases