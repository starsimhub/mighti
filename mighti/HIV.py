import numpy as np
import starsim as ss

__all__ = ['HIV']

class HIV(ss.Disease):

    def __init__(self, pars=None, **kwargs):
        # Parameters
        super().__init__()
        self.default_pars(
            beta=0.001,  # Overall transmission rate
            init_prev=ss.bernoulli(self.age_dependent_prevalence),  # Age-dependent initial prevalence
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('infected'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('ti_on_art'),
            ss.FloatArr('ti_dead'),
        )

        print("HIV initialized with age-dependent prevalence")

    def age_dependent_prevalence(self, module=None, sim=None, size=None):
        age_data = {
            0: 0,
            15: 0.056,
            20: 0.172,
            25: 0.303,
            30: 0.425,
            35: 0.525,
            40: 0.572,
            45: 0.501,
            50: 0.435,
            55: 0.338,
            60: 0.21,
            65: 0.147,
            99: 0,
        }
        n_age_bins = len(age_data) - 1
        age_bins = list(age_data.keys())
        ages = sim.people.age[size]  # Initial ages of agents
        prevalence = np.zeros(len(ages))
    
        print(f"Ages for agents (first 20): {ages[:20]}")  # Debug print
    
        for i in range(n_age_bins):
            left = age_bins[i]
            right = age_bins[i + 1]
            value = age_data[left]
            prevalence[(ages >= left) & (ages < right)] = value
    
        print(f"Prevalence initialized (first 20): {prevalence[:20]}")  # Debug print
    
        return prevalence

    def set_initial_states(self):
        initial_cases = self.pars['init_prev'].filter()
        self.infected[initial_cases] = True
        self.susceptible[~initial_cases] = True
    
        # Debug: Show how many are infected and a sample of who is infected
        print(f"Initial infection cases (infected): {np.sum(initial_cases)} out of {len(initial_cases)} agents")
        print(f"Infected agent sample: {self.infected[:20]}")  # Show a sample of infected agents
    
        return initial_cases

    def init_results(self):
        sim = self.sim
        super().init_results()
        self.results += [
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ss.Result(self.name, 'new_infections', sim.npts, dtype=int),
            ss.Result(self.name, 'cum_infections', sim.npts, dtype=int),
        ]
        return

    def update_results(self):
        sim = self.sim
        super().update_results()
        ti = sim.ti
        self.results.prevalence[ti] = np.count_nonzero(self.infected) / len(sim.people)
        self.results.new_infections[ti] = np.count_nonzero(self.ti_infected == ti)
        self.results.cum_infections[ti] = np.sum(self.results.new_infections[:ti+1])
        return