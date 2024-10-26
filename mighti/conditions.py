import numpy as np
import starsim as ss
import mighti as mi
import pandas as pd

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


mortality_data = pd.read_csv('mighti/data/mortality_risk.csv')
# Convert the Age column to a string to maintain consistency
mortality_data['Age'] = mortality_data['Age'].astype(str)

# Create a dictionary to store rates by condition, age, sex, and category
mortality_rates = {}
for _, row in mortality_data.iterrows():
    condition, age, sex, rate, category = row
    mortality_rates.setdefault(condition.strip(), {}).setdefault(age, {}).setdefault(sex, {})[category] = rate


# Helper function for retrieving mortality rates
def get_mortality_rate(condition, age, sex, category, mortality_data):
    try:
        return mortality_data[condition][age][sex][category]
    except KeyError:
        return 0  # Default to 0 if no data is available
    
    

class BaseNCD(ss.NCD):
    """Base class for NCDs with shared `make_new_cases` logic."""

    def make_new_cases(self, relative_risk=1.0):
        """Create new cases of NCDs, adjusted by relative risk and susceptibility from multiple interactions."""
        
        sim = self.sim
        
        # Get susceptible individuals
        susceptible_uids = self.susceptible.uids

        # Base incidence probability
        base_prob = self.pars.incidence_prob  # Use the stored probability

        # Combine relative susceptibility from both HIV, HPV, Flu, and other NCD interactions
        rel_sus_infectious = self.rel_sus[susceptible_uids]  # Start with base relative susceptibility
        rel_sus_ncd = np.ones_like(rel_sus_infectious)  # Start with neutral susceptibility (1.0)

        # Loop through diseases to apply infectious disease and NCD interactions
        for condition, cond_obj in sim.diseases.items():
            if condition in ['hiv', 'hpv', 'flu']:  # Handle infectious diseases
                infected_uids = np.intersect1d(susceptible_uids, cond_obj.infected.uids)
                rel_sus_infectious[np.isin(susceptible_uids, infected_uids)] *= cond_obj.rel_sus.raw[infected_uids]  # Apply rel_sus
            elif condition != self.name.lower():  # Skip self and apply only for NCDs
                affected_uids = np.intersect1d(susceptible_uids, cond_obj.affected.uids)
                rel_sus_ncd[np.isin(susceptible_uids, affected_uids)] *= cond_obj.rel_sus.raw[affected_uids]  # Apply rel_sus for NCDs

        # Combine the susceptibility effects from infectious diseases and NCDs
        combined_rel_sus = rel_sus_infectious * rel_sus_ncd

        # Adjust the incidence probability by the combined relative susceptibility
        adjusted_prob = base_prob * combined_rel_sus

        # Apply the adjusted incidence probability to susceptible individuals
        adjusted_incidence_dist = ss.bernoulli(adjusted_prob, strict=False)
        adjusted_incidence_dist.initialize()

        # Determine new cases based on the adjusted probability
        new_cases = adjusted_incidence_dist.rvs(len(susceptible_uids))  # Generate new cases
        new_cases = susceptible_uids[new_cases]  # Select new cases based on generated values

        # Set prognoses for new cases
        self.set_prognoses(new_cases)

        return new_cases

    
class Type1Diabetes(ss.NCD):
    
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_condition=ss.lognorm_ex(1),  # Shorter duration before serious complications
            incidence=ss.bernoulli(0.000015),      # Lower incidence of Type 1 diabetes
            p_death=ss.bernoulli(0.0033),        # Higher mortality rate from Type 1
            init_prev=ss.bernoulli(0.01),      # Initial prevalence of Type 1 diabetes
        )
        self.update_pars(pars, **kwargs)
        
        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus'),
        )
        return
    
    def initialize(self, sim):
        """Initialize the disease, setting rel_sus for each agent."""
        super().initialize(sim)
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
        # Identify which individuals have died at the current timestep
        deaths = (self.ti_dead == sim.ti).uids
        
        # Debugging print statement
        print(f"UIDs of deaths: {deaths}, Type: {type(deaths)}")
        
        if deaths.size == 0:
            print(f"No deaths at timestep {sim.ti}")
            return  # If no deaths, skip processing further
        
        # Ensure deaths are in the correct format (convert to a list of integers if needed)
        deaths = deaths.tolist() if isinstance(deaths, np.ndarray) else deaths
        print(f"Converted UIDs of deaths: {deaths}, Type: {type(deaths)}")
    
        # Log the deaths (only if deaths are present)
        self.log.add_data(deaths, died=True)

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


class Type2Diabetes(BaseNCD):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_condition=ss.lognorm_ex(5),  # Longer duration reflecting chronic condition
            incidence_prob = 0.0315,
            incidence=ss.bernoulli(0.0315),    # Higher incidence rate
            p_death=ss.bernoulli(0.0017),     # Mortality risk
            init_prev=ss.bernoulli(0.2),     # Higher initial prevalence
            remission_rate=ss.bernoulli(0.0024),  # Probability of remission
            max_disease_duration=20,         # Maximum duration before severe complications
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.BoolArr('reversed'),  # New state for diabetes remission
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus'),
        )
        return
    
    def init_post(self):
        """Initialize disease after simulation starts."""
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def update_pre(self):
        """Updates before each simulation step, such as handling remission and death."""
        sim = self.sim

        # Handle remission (reversal)
        going_into_remission = self.pars.remission_rate.filter(self.affected.uids)
        self.affected[going_into_remission] = False
        self.reversed[going_into_remission] = True
        self.ti_reversed[going_into_remission] = sim.ti

        # Handle recovery and deaths
        recovered = (self.reversed & (self.ti_reversed <= sim.ti)).uids
        self.reversed[recovered] = False
        self.susceptible[recovered] = True  # Recovered individuals become susceptible again
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)
        
        return

    def set_prognoses(self, uids):
        """Set the prognoses for new cases."""
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

        return

    def init_results(self):
        """Initialize the results for Type2Diabetes."""
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
        """Update the results for each timestep."""
        sim = self.sim
        super().update_results()
        self.results.prevalence[sim.ti] = np.count_nonzero(self.affected) / len(sim.people)
        self.results.reversal_prevalence[sim.ti] = np.count_nonzero(self.reversed) / len(sim.people)
        return
    

class Obesity(ss.NCD):
    
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_condition=ss.lognorm_ex(1),       # Duration of obesity condition
            incidence=ss.bernoulli(0.15),         # Incidence rate of obesity
            init_prev=ss.bernoulli(0.25),         # Initial prevalence of obesity
        )
        self.update_pars(pars, **kwargs)
        
        # States specific to obesity
        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('rel_sus'),
        )
        return
    
    def initialize(self, sim):
        """Initialize the disease, setting rel_sus for each agent."""
        super().initialize(sim)
        # Set initial relative susceptibility for interactions, defaulting to 1 (neutral)
        self.rel_sus[:] = 1.0  
        return

    def init_post(self):
        """Initialize agents affected by obesity at the start of the simulation."""
        initial_cases = self.pars.init_prev.filter()   # Filter agents based on initial prevalence
        self.set_prognoses(initial_cases)              # Set prognoses for these initial cases
        return initial_cases

    def update_pre(self):
        """Update states at each timestep, managing recovery."""
        sim = self.sim

        # Process recoveries based on recovery time
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True  # Recovered agents become susceptible again
        return

    def set_prognoses(self, uids):
        """Set prognoses for newly affected agents."""
        sim = self.sim
        self.susceptible[uids] = False  # Mark agents as affected
        self.affected[uids] = True

        # Set recovery times for the affected agents
        dur_condition = self.pars.dur_condition.rvs(uids)
        self.ti_recovered[uids] = sim.ti + dur_condition / sim.dt
        return

    def init_results(self):
        """Initialize results arrays for tracking prevalence and incidence."""
        sim = self.sim
        super().init_results()
        
        # Prevalence tracking
        if 'prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ]
        return

    def update_results(self):
        """Update results each timestep to track prevalence."""
        sim = self.sim
        super().update_results()
        # Calculate prevalence as the ratio of affected individuals to the population size
        self.results.prevalence[sim.ti] = np.count_nonzero(self.affected) / len(sim.people)
        return

class Hypertension(ss.NCD):
    
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_condition=ss.lognorm_ex(1),      # Duration of hypertension condition
            incidence=ss.bernoulli(0.276),        # Incidence rate of hypertension
            p_death=ss.bernoulli(0.001),         # Mortality risk due to hypertension
            init_prev=ss.bernoulli(0.5),        # Initial prevalence of hypertension
        )
        self.update_pars(pars, **kwargs)
        
        # Define states for hypertension
        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus'),
        )
        return

    def init_post(self):
        """Initialize agents affected by hypertension at the start of the simulation."""
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def update_pre(self):
        """Update states at each timestep, managing recovery and death."""
        sim = self.sim

        # Process recoveries based on recovery time
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True  # Recovered agents become susceptible again

        # Process deaths based on death time
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)

        return

    def make_new_cases(self):
        """Create new cases of hypertension among susceptible individuals."""
        new_cases = self.pars.incidence.filter(self.susceptible.uids)
        self.set_prognoses(new_cases)
        return new_cases

    def set_prognoses(self, uids):
        """Set prognoses for newly affected individuals."""
        sim = self.sim
        p = self.pars

        # Update states for the new cases
        self.susceptible[uids] = False
        self.affected[uids] = True
        dur_condition = p.dur_condition.rvs(uids)
        
        # Determine recovery or death for each new case
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]

        # Assign recovery or death time based on condition duration
        self.ti_dead[dead_uids] = sim.ti + dur_condition[will_die] / sim.dt
        self.ti_recovered[rec_uids] = sim.ti + dur_condition[~will_die] / sim.dt
        return

    def init_results(self):
        """Initialize results arrays for tracking prevalence and deaths."""
        sim = self.sim
        super().init_results()
        
        # Initialize prevalence tracking
        if 'prevalence' not in self.results:
            self.results += [
                ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ]
        # Initialize new deaths tracking
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
            ]
        return

    def update_results(self):
        """Update prevalence and death results at each timestep."""
        sim = self.sim
        super().update_results()
        
        # Calculate prevalence as the fraction of affected individuals
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
        self.update_pars(pars, **kwargs)
        
        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus'),
        )
        return
    

class Flu(ss.SIS):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            init_prev=ss.bernoulli(0.1),  # Example initial prevalence
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('infected'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('rel_sus'),  # Add relative susceptibility
        )
    # """
    # Example influenza model. Modifies the SIS model by adding a probability of dying.
    # Death probabilities are based on age.
    # """
    # def __init__(self, pars=None, **kwargs):
    #     super().__init__()
    #     self.default_pars(
    #         p_death=0,  # Placeholder - see make_p_death_fn
    #         dur_inf=ss.lognorm_ex(10),
    #         beta=0.05,
    #         init_prev=ss.bernoulli(0.01),
    #         waning=0.05,
    #         imm_boost=1.0,
    #     )
    #     self.update_pars(pars, **kwargs)
    #     self.add_states(
    #         ss.FloatArr('ti_dead'),
    #         ss.FloatArr('rel_sus'),
    #     )
    #     self.pars.p_death = ss.bernoulli(self.make_p_death_fn)

    #     return

    # @staticmethod
    # def make_p_death_fn(self, sim, uids):
    #     """ Take in the module, sim, and uids, and return the death probability for each UID based on their age """
    #     return mi.make_p_death_fn(name='flu', sim=sim, uids=uids)

    # def update_pre(self, sim):

    #     # Process people who recover and become susceptible again
    #     recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids
    #     self.infected[recovered] = False
    #     self.susceptible[recovered] = True
    #     self.update_immunity(sim)

    #     # Trigger deaths
    #     deaths = (self.ti_dead <= sim.ti).uids
    #     if len(deaths):
    #         sim.people.request_death(sim, deaths)

    #     return

    # def set_prognoses(self, sim, uids, source_uids=None):
    #     """ Set prognoses """
    #     self.susceptible[uids] = False
    #     self.infected[uids] = True
    #     self.ti_infected[uids] = sim.ti
    #     self.immunity[uids] += self.pars.imm_boost

    #     p = self.pars

    #     # Sample duration of infection, being careful to only sample from the
    #     # distribution once per timestep.
    #     dur_inf = p.dur_inf.rvs(uids)

    #     # Determine who dies and who recovers and when
    #     will_die = p.p_death.rvs(uids)
    #     dead_uids = uids[will_die]
    #     rec_uids = uids[~will_die]
    #     self.ti_dead[dead_uids] = sim.ti + dur_inf[will_die] / sim.dt # Consider rand round, but not CRN safe
    #     self.ti_recovered[rec_uids] = sim.ti + dur_inf[~will_die] / sim.dt

    #     return


### Defining only minimal
class HPV(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            init_prev=ss.bernoulli(0.1),  # Example initial prevalence
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('infected'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('rel_sus'),  # Add relative susceptibility
        )

class CervicalCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            init_prev=ss.bernoulli(0.05),
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class ColorectalCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class BreastCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class LungCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.04))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class ProstateCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class OtherCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class Parkinsons(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class Smoking(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.3))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class BRCA(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.005))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class Alcohol(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.15))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )
class ViralHepatitis(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )

class Poverty(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.4))
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.BoolArr('susceptible'), 
            ss.BoolArr('affected'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus'),
        )
        
class Accident(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.02))  
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),  
        )

class Alzheimers(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),
        )

class Assault(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.005))
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),
        )

class CerebrovascularDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),
        )

class ChronicLiverDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),
        )

class ChronicLowerRespiratoryDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),
        )

class HeartDisease(ss.NCD):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.05))
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),
        )

class ChronicKidneyDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),
            ss.BoolArr('affected'),
            ss.FloatArr('rel_sus'),
        )