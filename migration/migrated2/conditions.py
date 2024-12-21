
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
            incidence=ss.bernoulli(0.000015),  # Lower incidence of Type 1 diabetes
            p_death=ss.bernoulli(0.0033),  # Higher mortality rate from Type 1
            init_prev=ss.bernoulli(0.01),  # Initial prevalence of Type 1 diabetes
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus'),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
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
        self.ti_dead[dead_uids] = sim.ti + dur_condition[will_die] / sim.t.dt
        self.ti_recovered[rec_uids] = sim.ti + dur_condition[~will_die] / sim.t.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        if 'prevalence' not in self.results:
            self.results += [
                ss.Result('prevalence', sim.t.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result('new_deaths', sim.t.npts, dtype=int),
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
            ss.FloatArr('rel_sus'),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
        sim = self.sim

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

        deaths = (self.ti_dead == sim.ti).uids

        # Debugging print statement
        print(f"UIDs of deaths: {deaths}")

        # Check if deaths is a valid list or array
        if isinstance(deaths, np.ndarray):
            print(f"Deaths array shape: {deaths.shape}, type: {deaths.dtype}")
        else:
            print(f"Deaths type: {type(deaths)}")

        # Continue with logging
        self.log.add_data(deaths, died=True)
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
        self.ti_dead[dead_uids] = sim.ti + dur_condition[will_die] / sim.t.dt
        self.ti_reversed[rec_uids] = sim.ti + dur_condition[~will_die] / sim.t.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        if 'prevalence' not in self.results:
            self.results += [
                ss.Result('prevalence', sim.t.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result('new_deaths', sim.t.npts, dtype=int),
            ]
        if 'reversal_prevalence' not in self.results:
            self.results += [
                ss.Result('reversal_prevalence', sim.t.npts, dtype=float),
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
            ss.FloatArr('rel_sus'),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
        sim = self.sim
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
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
        self.ti_recovered[uids] = sim.ti + dur_condition / sim.t.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        # Check if the keys already exist in the results
        if 'prevalence' not in self.results:
            self.results += [
                ss.Result('prevalence', sim.t.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result('new_deaths', sim.t.npts, dtype=int),
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
            ss.FloatArr('rel_sus'),
        )
        return

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
        sim = self.sim
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)

        deaths = (self.ti_dead == sim.ti).uids

        # Debugging print statement
        print(f"UIDs of deaths: {deaths}")

        # Check if deaths is a valid list or array
        if isinstance(deaths, np.ndarray):
            print(f"Deaths array shape: {deaths.shape}, type: {deaths.dtype}")
        else:
            print(f"Deaths type: {type(deaths)}")

        # Continue with logging
        self.log.add_data(deaths, died=True)
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
        self.ti_dead[dead_uids] = sim.ti + dur_condition[will_die] / sim.t.dt
        self.ti_recovered[rec_uids] = sim.ti + dur_condition[~will_die] / sim.t.dt
        return

    def init_results(self):
        sim = self.sim
        super().init_results()

        # Check if the keys already exist in the results
        if 'prevalence' not in self.results:
            self.results += [
                ss.Result('prevalence', sim.t.npts, dtype=float),
            ]
        if 'new_deaths' not in self.results:
            self.results += [
                ss.Result('new_deaths', sim.t.npts, dtype=int),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),  # Add relative susceptibility
        )


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
            ss.FloatArr('rel_sus'),  # Add relative susceptibility
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
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
            ss.FloatArr('rel_sus'),
        )


class Accident(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )


class Alzheimers(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )


class Assault(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.005))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )


class CerebrovascularDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )


class ChronicLiverDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )


class ChronicLowerRespiratoryDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )


class HeartDisease(ss.NCD):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.05))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )


class ChronicKidneyDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible'),
            ss.State('affected'),
            ss.FloatArr('rel_sus'),
        )
