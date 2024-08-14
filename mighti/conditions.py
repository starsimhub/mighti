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
    'Accident', 'Alzheimers', 'Cerebro', 'Liver', 'Resp', 'Diabetes', 'Heart', 'Kidney', 'Flu',
    'Depression', 'HPV', 'Colorectal', 'Breast', 'Lung', 'Prostate', 'Other', 'Parkinsons',
    'HIV', 'Smoking', 'Obesity', 'Alcohol', 'BRCA',
]


# INDIVIDUAL CONDITIONS
class Accident(ss.Disease):
    pass


class Alzheimers(ss.Disease):
    pass


class Assault(ss.Disease):
    pass


class Cerebro(ss.Disease):
    pass


class Liver(ss.Disease):
    pass


class Resp(ss.Disease):
    pass


class Diabetes(ss.NCD):
    pass


class Heart(ss.NCD):
    pass


class Kidney(ss.Disease):
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


class Depression(ss.Disease):

    def __init__(self, pars=None, **kwargs):
        # Parameters
        super().__init__()
        self.default_pars(
            # Initial conditions
            dur_episode=ss.lognorm_ex(1),  # Duration of an episode
            incidence=ss.bernoulli(0.1),  # Incidence at each point in time
            p_death=ss.bernoulli(0.001),  # Risk of death from depression (e.g. by suicide)
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
        """
        new_cases = self.pars['incidence'].filter()
        self.affected[new_cases] = True
        return new_cases

    def update_pre(self, sim):
        recovered = (self.affected & (self.ti_recovered <= sim.ti)).uids
        self.affected[recovered] = False
        self.susceptible[recovered] = True
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(sim, deaths)
        self.results.new_deaths[sim.ti] = len(deaths)  # Log deaths attributable to this module
        return

    def make_new_cases(self, sim):
        new_cases = self.pars['incidence'].filter(self.susceptible.uids)
        self.set_prognoses(sim, new_cases)
        return

    def set_prognoses(self, sim, uids):
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

    def init_results(self, sim):
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
        ]
        return

    def update_results(self, sim):
        super().update_results(sim)
        ti = sim.ti
        self.results.prevalence[ti] = np.count_nonzero(self.affected)/len(sim.people)
        self.results.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        return


class HPV(ss.Disease):
    pass


class Colorectal(ss.Disease):
    pass


class Breast(ss.Disease):
    pass


class Lung(ss.Disease):
    pass


class Prostate(ss.Disease):
    pass


class Other(ss.Disease):
    pass


class Parkinsons(ss.Disease):
    pass


class HIV(ss.Disease):
    pass


class Smoking(ss.Disease):
    pass


class BRCA(ss.Disease):
    pass


class Obesity(ss.Disease):
    pass


class Alcohol(ss.Disease):
    pass
