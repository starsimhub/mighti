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
    'Depression','AlzheimersDisease', 'ParkinsonsDisease','PTSD','HIVAssociatedDementia',
    'CardiovascularDisease','ChronicLiverDisease','Asthma',
    'RoadInjury','DomesticViolence','TobaccoUse', 'AlcoholUseDisorder',
    'ChronicKidneyDisease','Flu','HPVVaccination',
    'ViralHepatitis','COPD','Hyperlipidemia',
    'CervicalCancer','ColorectalCancer', 'BreastCancer', 'LungCancer', 'ProstateCancer', 'OtherCancer',
]

# __all__ = [
#     'Type2Diabetes', 'Obesity'
# ]



class Type2Diabetes(ss.NCD):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            dur_condition=ss.lognorm_ex(15.02897096),
            incidence_prob=0.010125, 
            incidence=ss.bernoulli(0.010125),
            p_death=ss.bernoulli(0.004315),
            init_prev=ss.bernoulli(0.1351),
            remission_rate=ss.bernoulli(0.024),
            max_disease_duration=30,
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
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
        # print(f"\n Debugging `init_prev` in Type2Diabetes: {self.pars.init_prev}")

        initial_cases = self.pars.init_prev.filter()

        print(f"Expected initial cases: {len(initial_cases)}")  # Should be around 13.5% of n_agents

        if len(initial_cases) == 0:
            print("WARNING: `init_prev` is filtering 0 cases! Something is wrong.")

        self.set_prognoses(initial_cases)

        #  Debugging After Initialization
        # print(f" After `init_post`:")
        # print(f"  - Affected (should be ~67,651): {np.sum(self.affected.raw)}")
        # print(f"  - Susceptible: {np.sum(self.susceptible.raw)}")

        return initial_cases

    def step_state(self):
        # Handle remission (reversal)
        going_into_remission = self.pars.remission_rate.filter(self.affected.uids)
        self.affected[going_into_remission] = False
        self.reversed[going_into_remission] = True
        self.ti_reversed[going_into_remission] = self.ti

        # Handle recovery & deaths
        recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
        self.reversed[recovered] = False
        self.susceptible[recovered] = True  # Recovered individuals become susceptible again
        deaths = (self.ti_dead == self.ti).uids
        self.sim.people.request_death(deaths)

    def step(self):
        # print(f"\n Debugging `step` in {self.name}:")
        # print(f"  - Susceptible count before new cases: {np.sum(self.susceptible.raw)}")

        # Generate new cases
        new_cases = self.pars.incidence.filter(self.susceptible.uids)

        # print(f"  - New cases detected: {len(new_cases)}")

        self.set_prognoses(new_cases)

        # print(f"  - Affected count after step: {np.sum(self.affected.raw)}")

        return new_cases


    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
        # print(f"Debugging `set_prognoses` for {len(uids)} cases.")  # Should be ~67,651

        if len(uids) == 0:
            print("WARNING: No affected cases! This may be an issue.")



        self.susceptible[uids] = False
        self.affected[uids] = True
        # print(f" Affected count after assignment: {np.sum(self.affected.raw)}")  # Expect ~67,651

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
            ss.Result('prevalence_in_plhiv', dtype=float),  # Initialize this key
            ss.Result('prevalence_in_hivneg', dtype=float),  # Initialize this key
            ss.Result('reversal_prevalence', dtype=float),
        )
        return

    def update_results(self):
        super().update_results()

        # Compute overall T2D prevalence
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.reversal_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

        # Identify PLHIV and HIV-negative individuals
        plhiv = self.sim.people.hiv.infected  # Ensure this correctly refers to HIV-positive individuals
        hiv_negative = ~plhiv

        plhiv_count = np.count_nonzero(plhiv)
        hiv_negative_count = np.count_nonzero(hiv_negative)

        # Compute T2D prevalence in PLHIV and HIV-negative groups
        self.results.setdefault('prevalence_in_plhiv', np.full(len(self.sim.results.timevec), np.nan))
        self.results.setdefault('prevalence_in_hivneg', np.full(len(self.sim.results.timevec), np.nan))

        if plhiv_count > 0:
            self.results.prevalence_in_plhiv[self.ti] = np.count_nonzero(self.affected & plhiv) / plhiv_count
        else:
            self.results.prevalence_in_plhiv[self.ti] = np.nan

        if hiv_negative_count > 0:
            self.results.prevalence_in_hivneg[self.ti] = np.count_nonzero(self.affected & hiv_negative) / hiv_negative_count
        else:
            self.results.prevalence_in_hivneg[self.ti] = np.nan

        # Define age groups and sexes
        age_groups = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        age_group_labels = [f"{age_groups[i]}_{age_groups[i+1]}" for i in range(len(age_groups) - 1)]
        sexes = ["male", "female"]

        # Ensure sex is correctly interpreted
        male = ~self.sim.people.female  # Assuming `female` is a boolean array
        female = self.sim.people.female

        # Compute and store T2D prevalence by age and sex
        for sex, is_sex in zip(sexes, [male, female]):
            for i, label in enumerate(age_group_labels):
                age_min, age_max = age_groups[i], age_groups[i+1]
                in_age_group = (self.sim.people.age >= age_min) & (self.sim.people.age < age_max)
                in_group = in_age_group & is_sex

                total_in_group = np.count_nonzero(in_group)

                # Initialize results storage if it does not exist
                key = f'T2D_prevalence_{sex}_{label}'
                if key not in self.results:
                    self.results[key] = np.full(len(self.sim.results.timevec), np.nan)

                # Compute prevalence
                if total_in_group > 0:
                    self.results[key][self.ti] = np.count_nonzero(self.affected & in_group) / total_in_group
                else:
                    self.results[key][self.ti] = np.nan


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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
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


class Hyperlipidemia(ss.NCD):
    

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
            ss.State('susceptible', default=True),
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

class Asthma(ss.NCD):

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
            ss.State('susceptible', default=True),
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


class COPD(ss.NCD):

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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        )
        return
  

class PTSD(ss.Disease):

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
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        )
        return
    
    
class HIVAssociatedDementia(ss.Disease):

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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
            ss.State('infected'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('rel_sus', default=1.0),
        )
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
            ss.FloatArr('rel_sus', default=1.0),
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


class HPVVaccination(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            init_prev=ss.bernoulli(0.1),  # Example initial prevalence
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ParkinsonsDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class TobaccoUse(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.3))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class AlcoholUseDisorder(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.15))
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.State('susceptible', default=True),
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
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class RoadInjury(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class DomesticViolence(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )
        

class AlzheimersDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.01))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class CardiovascularDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ChronicLiverDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.02))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class HeartDisease(ss.NCD):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=ss.bernoulli(0.05))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )


class ChronicKidneyDisease(ss.Disease):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(init_prev=ss.bernoulli(0.03))
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('affected'),
            ss.FloatArr('rel_sus', default=1.0),
        )
