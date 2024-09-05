import numpy as np
import starsim as ss
import mighti as mi

# CONDITIONS
# This is an umbrella term for any health condition. Some conditions can lead directly
# to death/disutility (e.g. heart disease, HIV, depression), while others do not. All
# conditions can affect the (1) risk of acquiring, (2) persistence of, (3) severity of
# other conditions.
# Classes:
#    - BaseCondition (Base),
#    - PermanentRisk (death no, recovery no. Examples: genetic mutations)
#    - RemovableRisk (death no, recovery yes. Examples: obesity)
#    - FatalCondition (death yes, recovery no; examples: HIV, alzheimers)
#    - RecoverableCondition (death yes, recovery yes; examples: depression)
# Some examples:
#    - HIV increases likelihood of getting depression & of depression persisting
#    - depression increases likelihood of getting HIV

# Specify all externally visible classes this file defines
__all__ = [
    'Depression',
]




class Depression(ss.Disease):

    def __init__(self, pars=None, **kwargs):
        # Parameters
        super().__init__()
        self.default_pars(
            # Initial conditions
            dur_episode=ss.lognorm_ex(1),  # Duration of an episode
            incidence=ss.bernoulli(0.9),  # Incidence at each point in time
            p_death=ss.bernoulli(0.001),  # Risk of death from depression (e.g. by suicide)
            init_prev=ss.bernoulli(0.0),  # Default initial prevalence (modified below for age-dependent prevalence)
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
        )

        return


    def set_initial_states(self, sim):
        """
        Set the initial states of the population based on the age-dependent initial prevalence.
        """
        # Create an array to hold initial prevalence values for each person
        initial_prevalence = np.zeros(len(sim.people))
        
        # Loop through each individual and sample the initial prevalence
        for i in range(len(sim.people)):
            initial_prevalence[i] = self.pars['init_prev'].rvs()  # Sample individually
        
        # Set the initial 'affected' state based on the prevalence function
        self.affected = np.random.random(len(sim.people)) < initial_prevalence  # Apply age-dependent prevalence
        print(f"Initial affected individuals: {np.sum(self.affected)}")  # Debug: Print number of initially affected individuals
        return initial_prevalence


    def update_pre(self):
        sim = self.sim
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)  # Log deaths attributable to this module
        return

    
    def make_new_cases(self):
        # Generate new depression cases among susceptible people
        new_cases = self.pars['incidence'].filter(self.susceptible.uids)  # Find new cases based on incidence rate
       # print(f"Time {self.sim.ti}: New depression cases: {len(new_cases)}")  # Debug: Print number of new cases
        self.set_prognoses(new_cases)  # Assign prognoses to the new cases
        return new_cases

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
        self.affected[uids] = True

        # Sample duration of episode
        dur_ep = p.dur_episode.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = sim.ti + dur_ep[will_die] / sim.dt
        self.ti_recovered[rec_uids] = sim.ti + dur_ep[~will_die] / sim.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()
        self.results += [
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
        ]
        return

    def update_results(self):
        sim = self.sim
        super().update_results()
        ti = sim.ti
        age_groups = sim.people.age  # Access people's ages
        
        # Overall prevalence
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        
        # Age group bins
        age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 99]
        
        # Prevalence by age group
        prevalence_by_age = np.zeros(len(age_bins) - 1)
        
        # Loop through age groups and calculate prevalence
        for i, (left, right) in enumerate(zip(age_bins[:-1], age_bins[1:])):
            age_mask = (age_groups >= left) & (age_groups < right)
            if np.count_nonzero(age_mask) > 0:
                prevalence_by_age[i] = np.count_nonzero(self.affected[age_mask]) / np.count_nonzero(age_mask)
        
        #print(f"Prevalence by age group at time {ti}: {prevalence_by_age}")  # Debug print
        self.results.prevalence_by_age = prevalence_by_age  # Store prevalence by age group
        return