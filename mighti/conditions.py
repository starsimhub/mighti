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
    'Depression', 'Accident', 'Alzheimers', 'Assault', 'CerebrovascularDisease',
    'ChronicLiverDisease', 'ChronicLowerRespiratoryDisease', 'HeartDisease',
    'ChronicKidneyDisease', 'Flu', 'HPV',
    'CervicalCancer', 'ColorectalCancer', 'BreastCancer', 'LungCancer', 'ProstateCancer', 'OtherCancer',
    'Parkinsons', 'Smoking', 'Alcohol', 'BRCA', 'ViralHepatitis', 'Poverty'
]



class Type1Diabetes(ss.NCD):
    
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            dur_condition=ss.lognorm_ex(1),  # Shorter duration before serious complications
            incidence=ss.bernoulli(0.000015),      # Lower incidence of Type 1 diabetes
            p_death=ss.bernoulli(0.0033),        # Higher mortality rate from Type 1
            init_prev=ss.bernoulli(0.01),      # Initial prevalence of Type 1 diabetes
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
        recovered = (self.affected & (self.ti_recovered <= self.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        deaths = (self.ti_dead == self.ti).uids
        self.sim.people.request_death(deaths)
        self.results.new_deaths[self.ti] = len(deaths)
        # self.log.add_data(deaths, died=True)
        return

    def step(self):
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
        self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt # CK: TODO: fix
        self.ti_recovered[rec_uids] = self.ti + dur_condition[~will_die] / self.t.dt
        return

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return


class Type2Diabetes(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            dur_condition=ss.lognorm_ex(5),  # Longer duration reflecting chronic condition
            incidence_prob=0.0315,
            incidence=ss.bernoulli(0.0315),  # Higher incidence rate
            p_death=ss.bernoulli(0.0017),  # Mortality risk (may increase over time)
            init_prev=ss.bernoulli(0.2),  # Higher initial prevalence
            remission_rate=ss.bernoulli(0.0024),  # Probability of remission (reversing the condition)
            max_disease_duration=20,  # Maximum duration before severe complications
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.State('reversed'),  # New state for diabetes remission
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            # ss.FloatArr('beta_cell_function'),  # Tracks beta-cell function over time
            # ss.FloatArr('insulin_resistance'),  # Tracks insulin resistance progression
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        # Initialize beta-cell function and insulin resistance
        # self.beta_cell_function[initial_cases] = 1.0  # Full function at the start
        # self.insulin_resistance[initial_cases] = 0.0  # No resistance initially
        return initial_cases

    def step_state(self):
        # Gradually increase insulin resistance and decrease beta-cell function
        # self.insulin_resistance[self.affected] += self.pars.insulin_resistance_increase_rate * sim.dt
        # self.beta_cell_function[self.affected] -= self.pars.beta_cell_decline_rate * sim.dt

        # Handle remission (reversal)
        going_into_remission = self.pars.remission_rate.filter(self.affected.uids)
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
        # self.log.add_data(deaths, died=True)

    def step(self):
        new_cases = self.pars.incidence.filter(self.susceptible.uids)
        self.set_prognoses(new_cases)
        return new_cases
    
    # def make_new_cases(self, relative_risk=1.0):
    #     """Create new cases of Type 2 Diabetes, adjusted by relative risk and susceptibility from multiple interactions."""
    
    #     sim = self.sim
    
    #     # Get susceptible individuals
    #     susceptible_uids = self.susceptible.uids
    
    #     # Base incidence probability
    #     base_prob = self.pars.incidence_prob  # Use the stored probability
    
    #     # Combine relative susceptibility from both HIV and other NCD interactions
    #     rel_sus_hiv = self.rel_sus[susceptible_uids]  # HIV-related relative susceptibility
    #     rel_sus_ncd = np.ones_like(rel_sus_hiv)  # Start with a neutral susceptibility (1.0)
    
    #     # Loop through other diseases in the simulation to apply NCD-NCD interactions
    #     for condition, cond_obj in sim.diseases.items():
    #         if condition == 'hiv':  # Special case for HIV
    #             # Only apply to those in susceptible_uids who are infected with HIV
    #             infected_uids = np.intersect1d(susceptible_uids, cond_obj.infected.uids)
    #             rel_sus_ncd[np.isin(susceptible_uids, infected_uids)] *= cond_obj.rel_sus.raw[infected_uids]  # Use .raw
    #         elif condition != 'type2diabetes':  # Skip self and apply only for NCDs
    #             # Only apply to those in susceptible_uids who are affected by another NCD
    #             affected_uids = np.intersect1d(susceptible_uids, cond_obj.affected.uids)
    #             rel_sus_ncd[np.isin(susceptible_uids, affected_uids)] *= cond_obj.rel_sus.raw[affected_uids]  # Use .raw
    
    #     # Combine the HIV susceptibility with the NCD susceptibilities
    #     combined_rel_sus = rel_sus_hiv * rel_sus_ncd
    
    #     # Adjust the incidence probability by the combined relative susceptibility
    #     adjusted_prob = base_prob * combined_rel_sus
    
    #     # Apply the adjusted incidence probability to susceptible individuals
    #     adjusted_incidence_dist = ss.bernoulli(adjusted_prob, strict=False)
    #     adjusted_incidence_dist.initialize()
    
    #     # Determine new cases based on the adjusted probability
    #     new_cases = adjusted_incidence_dist.rvs(len(susceptible_uids))  # Generate new cases
    #     new_cases = susceptible_uids[new_cases]  # Select new cases based on generated values
    
    #     # Set prognoses for new cases
    #     self.set_prognoses(new_cases)
    
    #     return new_cases
    

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
        self.susceptible[uids] = False
        self.affected[uids] = True
        dur_condition = p.dur_condition.rvs(uids)
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt
        self.ti_reversed[rec_uids] = self.ti + dur_condition[~will_die] / self.t.dt
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('reversal_prevalence', dtype=float),
        )
        return

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.reversal_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        return


class Obesity(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            dur_condition=ss.lognorm_ex(1),
            incidence=ss.bernoulli(0.15),
            init_prev=ss.bernoulli(0.25),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('rel_sus', default=1.0),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
        recovered = (self.affected & (self.ti_recovered <= self.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        return

    def step(self):
        new_cases = self.pars.incidence.filter(self.susceptible.uids)
        self.set_prognoses(new_cases)
        return new_cases

    def set_prognoses(self, uids):
        sim = self.sim
        self.susceptible[uids] = False
        self.affected[uids] = True
        dur_condition = self.pars.dur_condition.rvs(uids)
        self.ti_recovered[uids] = self.ti + dur_condition / self.t.dt
        return

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return


class Hypertension(ss.NCD):
    
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            dur_condition=ss.lognorm_ex(1),
            incidence=ss.bernoulli(0.12),
            p_death=ss.bernoulli(0.001),
            init_prev=ss.bernoulli(0.18),
        )
        self.update_pars(pars, **kwargs)
        
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
        recovered = (self.affected & (self.ti_recovered <= self.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        deaths = (self.ti_dead == self.ti).uids
        self.sim.people.request_death(deaths)
        self.results.new_deaths[self.ti] = len(deaths)
        # self.log.add_data(deaths, died=True)
        return

    def step(self):
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
        self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt
        self.ti_recovered[rec_uids] = self.ti + dur_condition[~will_die] / self.t.dt
        return

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return


class Depression(ss.Disease):

    def __init__(self, pars=None, **kwargs):
        # Parameters
        super().__init__()
        self.define_pars(
            # Initial conditions
            dur_episode=ss.lognorm_ex(1),  # Duration of an episode
            incidence=ss.bernoulli(0.9),  # Incidence at each point in time
            p_death=ss.bernoulli(0.001),  # Risk of death from depression (e.g. by suicide)
            init_prev=ss.bernoulli(0.2),  # Default initial prevalence (modified below for age-dependent prevalence)
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        )
        return


class Flu(ss.SIS):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            init_prev=ss.bernoulli(0.1),  # Example initial prevalence
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('infected'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('rel_sus', default=1.0),
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
    #         ss.FloatArr('rel_sus', default=1.0),
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
        self.define_pars(
            init_prev=ss.bernoulli(0.1),  # Example initial prevalence
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('infected'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class CervicalCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            init_prev=ss.bernoulli(0.05),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ColorectalCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class BreastCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class LungCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.04))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ProstateCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class OtherCancer(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class Parkinsons(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class Smoking(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.3))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class BRCA(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.005))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class Alcohol(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.15))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ViralHepatitis(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class Poverty(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.4))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class Accident(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class Alzheimers(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class Assault(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.005))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class CerebrovascularDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ChronicLiverDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ChronicLowerRespiratoryDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class HeartDisease(ss.NCD):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.05))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ChronicKidneyDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )
