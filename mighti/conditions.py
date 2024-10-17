import numpy as np
import starsim as ss
import mighti as mi

# CONDITIONS
# This is an umbrella term for any health condition. Some conditions can lead directly
# to death/disutility (e.g. heart disease, HIV, depression), while others do not. All
# conditions can affect the (1) risk of acquiring, (2) persistence of, (3) severity of
# other conditions.



__all__ = [
    'Type1Diabetes', 'Type2Diabetes', 'Obesity', 'Hypertension',
    'Depression','Accident', 'Alzheimers', 'Assault', 'CerebrovascularDisease',
    'ChronicLiverDisease','ChronicLowerRespiratoryDisease', 'HeartDisease',
    'ChronicKidneyDisease','Flu','HPV',
    'CervicalCancer','ColorectalCancer', 'BreastCancer', 'LungCancer', 'ProstateCancer', 'OtherCancer',
    'Parkinsons','Smoking', 'Alcohol', 'BRCA', 'ViralHepatitis', 'Poverty'
]



class Type1Diabetes(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_condition=ss.lognorm_ex(1),  # Shorter duration before serious complications
            incidence=ss.bernoulli(0.000015),      # Lower incidence of Type 1 diabetes
            p_death=ss.bernoulli(0.0033),        # Higher mortality rate from Type 1
            init_prev=ss.bernoulli(0.01),      # Initial prevalence of Type 1 diabetes
        )
        self.rel_sus = None  # Initialize rel_sus to store relative susceptibility
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
        )
        return


    def initialize(self, sim):
        """Initialize the disease, setting rel_sus for each agent."""
        super().initialize(sim)
        self.rel_sus = np.ones(sim.n)  # Initialize rel_sus for each agent in the sim (default to 1.0)
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def update_pre(self):
        sim = self.sim
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)
        return

    def make_new_cases(self):
        new_cases = self.pars.incidence.filter(self.susceptible.uids)
        self.set_prognoses(new_cases)
        return new_cases

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
        self.susceptible[uids] = False
        self.affected[uids] = True
        dur_condition = p.dur_condition.rvs(uids)
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = sim.ti + dur_condition[will_die] / sim.dt
        self.ti_recovered[rec_uids] = sim.ti + dur_condition[~will_die] / sim.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        if 'prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
            ]

        return

    def update_results(self):
        sim = self.sim
        super().update_results()
        self.results.prevalence[sim.ti] = np.count_nonzero(self.affected) / len(sim.people)
        return


class Type2Diabetes(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.rel_sus = None  # Initialize rel_sus to store relative susceptibility

        self.default_pars(
            dur_condition=ss.lognorm_ex(5),  # Longer duration reflecting chronic condition
            incidence_prob = 0.0315,
            incidence=ss.bernoulli(0.0315),    # Higher incidence rate
            p_death=ss.bernoulli(0.0017),     # Mortality risk (may increase over time)
            init_prev=ss.bernoulli(0.2),     # Higher initial prevalence
            # beta_cell_decline_rate=0.05,     # Rate of beta-cell function decline over time
            # insulin_resistance_increase_rate=0.1,  # Rate of increasing insulin resistance
            remission_rate=ss.bernoulli(0.0024),  # Probability of remission (reversing the condition)
            max_disease_duration=20,         # Maximum duration before severe complications
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.BoolArr('reversed'),          # New state for diabetes remission
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'),
            # ss.FloatArr('beta_cell_function'),  # Tracks beta-cell function over time
            # ss.FloatArr('insulin_resistance'),  # Tracks insulin resistance progression
        )
        return

    def initialize(self, sim):
        """Initialize the disease, setting rel_sus for each agent."""
        print(f"Calling initialize for {self.name}")  # Add this print to confirm

        super().initialize(sim)
        self.rel_sus = np.ones(len(sim.people))  # Initialize rel_sus for each agent in the sim (default to 1.0)
        print(f"Initialized rel_sus for Type2Diabetes: {self.rel_sus}")  # Debugging statement

        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        # Initialize beta-cell function and insulin resistance
        # self.beta_cell_function[initial_cases] = 1.0  # Full function at the start
        # self.insulin_resistance[initial_cases] = 0.0  # No resistance initially
        return initial_cases

    def update_pre(self):
        sim = self.sim
        # Gradually increase insulin resistance and decrease beta-cell function
        # self.insulin_resistance[self.affected] += self.pars.insulin_resistance_increase_rate * sim.dt
        # self.beta_cell_function[self.affected] -= self.pars.beta_cell_decline_rate * sim.dt

        # Handle remission (reversal)
        going_into_remission = self.pars.remission_rate.filter(self.affected.uids)
        self.affected[going_into_remission] = False
        self.reversed[going_into_remission] = True
        self.ti_reversed[going_into_remission] = sim.ti

        # Handle recovery, death, and beta-cell function exhaustion
        recovered = (self.reversed & (self.ti_reversed <= sim.ti)).uids
        self.reversed[recovered] = False
        self.susceptible[recovered] = True  # Recovered individuals become susceptible again
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)

        # Check if beta-cell function has dropped too low, causing death or severe progression
        # low_beta_function = self.affected & (self.beta_cell_function < 0.2)  # Threshold for beta-cell exhaustion
        # sim.people.request_death(low_beta_function.uids)
        return


    def make_new_cases(self, relative_risk=1.0):
        """Create new cases of Type2Diabetes, adjusted by relative risk."""

        # Get susceptible individuals
        susceptible_uids = self.susceptible.uids

        # Adjust incidence based on relative risk
        base_prob = self.pars.incidence_prob  # Use the stored probability
        adjusted_prob = base_prob * relative_risk  # Apply relative risk adjustment

        # print(f"Adjusted probability: {adjusted_prob}")

        # Create a bernoulli distribution with the adjusted probability and initialize it
        adjusted_incidence_dist = ss.bernoulli(adjusted_prob, strict=False)
        adjusted_incidence_dist.initialize()  # Explicitly initialize the distribution

        # Filter based on the adjusted probability
        new_cases = adjusted_incidence_dist.rvs(len(susceptible_uids))  # Generate new cases
        new_cases = susceptible_uids[new_cases]  # Select new cases based on generated values

        # print(f"New cases after applying relative risk {relative_risk}: {len(new_cases)}")

        # Set prognoses for new cases
        self.set_prognoses(new_cases)

        return new_cases


    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
        self.susceptible[uids] = False
        self.affected[uids] = True
        dur_condition = p.dur_condition.rvs(uids)
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = sim.ti + dur_condition[will_die] / sim.dt
        self.ti_reversed[rec_uids] = sim.ti + dur_condition[~will_die] / sim.dt

        # Set initial insulin resistance and beta-cell function
        # self.insulin_resistance[uids] = 0.0  # Start with no insulin resistance
        # self.beta_cell_function[uids] = 1.0  # Full beta-cell function initially
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        if 'prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
            ]
        if 'reversal_prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'reversal_prevalence', sim.npts, dtype=float),
            ]
        return

    def update_results(self):
        sim = self.sim
        super().update_results()
        self.results.prevalence[sim.ti] = np.count_nonzero(self.affected) / len(sim.people)
        self.results.reversal_prevalence[sim.ti] = np.count_nonzero(self.reversed) / len(sim.people)
        return



class Obesity(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_condition=ss.lognorm_ex(1),
            incidence=ss.bernoulli(0.15),
            init_prev=ss.bernoulli(0.25),
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('rel_sus'),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def update_pre(self):
        sim = self.sim
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        return

    def make_new_cases(self):
        new_cases = self.pars.incidence.filter(self.susceptible.uids)
        self.set_prognoses(new_cases)
        return new_cases

    def set_prognoses(self, uids):
        sim = self.sim
        self.susceptible[uids] = False
        self.affected[uids] = True
        dur_condition = self.pars.dur_condition.rvs(uids)
        self.ti_recovered[uids] = sim.ti + dur_condition / sim.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        # Check if the keys already exist in the results
        if 'prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
            ]

        return

    def update_results(self):
        sim = self.sim
        super().update_results()
        self.results.prevalence[sim.ti] = np.count_nonzero(self.affected) / len(sim.people)
        return


class Hypertension(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_condition=ss.lognorm_ex(1),
            incidence=ss.bernoulli(0.12),
            p_death=ss.bernoulli(0.001),
            init_prev=ss.bernoulli(0.18),
        )
        self.rel_sus = None  # Initialize rel_sus to store relative susceptibility
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def update_pre(self):
        sim = self.sim
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)
        return

    def make_new_cases(self):
        new_cases = self.pars.incidence.filter(self.susceptible.uids)
        self.set_prognoses(new_cases)
        return new_cases

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
        self.susceptible[uids] = False
        self.affected[uids] = True
        dur_condition = p.dur_condition.rvs(uids)
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = sim.ti + dur_condition[will_die] / sim.dt
        self.ti_recovered[rec_uids] = sim.ti + dur_condition[~will_die] / sim.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        # Check if the keys already exist in the results
        if 'prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
            ]

        return

    def update_results(self):
        sim = self.sim
        super().update_results()
        self.results.prevalence[sim.ti] = np.count_nonzero(self.affected) / len(sim.people)
        return




class Depression(ss.Disease):

    def __init__(self, pars=None, **kwargs):
        # Parameters
        super().__init__()
        self.default_pars(
            # Initial conditions
            dur_episode=ss.lognorm_ex(1),  # Duration of an episode
            incidence=ss.bernoulli(0.9),  # Incidence at each point in time
            p_death=ss.bernoulli(0.001),  # Risk of death from depression (e.g. by suicide)
            init_prev=ss.bernoulli(0.2),  # Default initial prevalence (modified below for age-dependent prevalence)
        )
        self.rel_sus = None  # Initialize rel_sus to store relative susceptibility
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
        )

        return


    def init_post(self):
        """
        Set the initial states of the population based on the age-dependent initial prevalence.
        """
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)

        # print(f"Initial affected individuals: {np.sum(self.affected)}")  # Debug: Print number of initially affected individuals
        return initial_cases

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
        self.susceptible[uids] = False
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

        # Check if the keys already exist in the results
        if 'prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
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
        age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

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




# INDIVIDUAL CONDITIONS
class Accident(ss.Disease):
    pass


class Alzheimers(ss.Disease):
    pass


class Assault(ss.Disease):
    pass


class CerebrovascularDisease(ss.Disease):
    pass


class ChronicLiverDisease(ss.Disease):
    pass


class ChronicLowerRespiratoryDisease(ss.Disease):
    pass


class HeartDisease(ss.NCD):
    pass


class ChronicKidneyDisease(ss.Disease):
    pass


class Flu(ss.SIS):
    """
    Example influenza model. Modifies the SIS model by adding a probability of dying.
    Death probabilities are based on age.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            p_death=0,  # Placeholder - see make_p_death_fn
            dur_inf=ss.lognorm_ex(10),
            beta=0.05,
            init_prev=ss.bernoulli(0.01),
            waning=0.05,
            imm_boost=1.0,
        )
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.FloatArr('ti_dead'),
        )
        self.pars.p_death = ss.bernoulli(self.make_p_death_fn)

        return

    @staticmethod
    def make_p_death_fn(self, sim, uids):
        """ Take in the module, sim, and uids, and return the death probability for each UID based on their age """
        return mi.make_p_death_fn(name='flu', sim=sim, uids=uids)

    def update_pre(self, sim):

        # Process people who recover and become susceptible again
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity(sim)

        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids
        if len(deaths):
            sim.people.request_death(sim, deaths)

        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses """
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        self.immunity[uids] += self.pars.imm_boost

        p = self.pars

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = sim.ti + dur_inf[will_die] / sim.dt # Consider rand round, but not CRN safe
        self.ti_recovered[rec_uids] = sim.ti + dur_inf[~will_die] / sim.dt

        return


class HPV(ss.Disease):
    pass

class CervicalCancer(ss.Disease):
    pass


class ColorectalCancer(ss.Disease):
    pass


class BreastCancer(ss.Disease):
    pass


class LungCancer(ss.Disease):
    pass


class ProstateCancer(ss.Disease):
    pass


class OtherCancer(ss.Disease):
    pass


class Parkinsons(ss.Disease):
    pass


class Smoking(ss.Disease):
    pass


class BRCA(ss.Disease):
    pass


class Alcohol(ss.Disease):
    pass

class ViralHepatitis(ss.Disease):
    pass

class Poverty(ss.Disease):
    pass