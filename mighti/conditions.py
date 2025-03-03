import numpy as np
import pandas as pd
import starsim as ss


df_params = None  # Placeholder for external data
disease_classes = {}  # Placeholder for dynamically created disease classes

def initialize_conditions(data):
    """ 
    Initialize conditions with externally loaded parameter data and generate disease classes. 
    Ensure that the dynamically created classes are accessible in `mighti` namespace.
    """
    global df_params, disease_classes
    df_params = data

    if df_params is None or df_params.empty:
        raise ValueError("[ERROR] `df_params` is None or empty! Ensure `initialize_conditions(df_params)` is called in mighti_main.py.")

    # Now create disease classes dynamically *after* df_params is initialized
    disease_classes.update({
        disease_name.replace(" ", "").replace("-", ""): create_disease_class(disease_name)
        for disease_name in df_params.index
    })

    # Register these disease classes globally in mighti module
    import mighti
    mighti.__dict__.update(disease_classes)  # Explicitly add to mighti's namespace

def get_param(condition, param_name, default=None):
    """ Retrieve parameter value safely after initialization. """
    if df_params is None:
        raise ValueError("[ERROR] `df_params` has not been initialized. Call `initialize_conditions(df_params)` in mighti_main.py before using conditions.")

    try:
        value = df_params.loc[condition, param_name]
        return default if pd.isna(value) else float(value) if isinstance(value, (int, float)) else value
    except KeyError:
        return default


# =========================
#  BASE CLASS FOR NCDs
# =========================
class BaseNCD(ss.NCD):
    """ Base class for Non-Communicable Diseases (chronic/remitting). """
    def __init__(self, condition, pars=None, **kwargs):
        super().__init__()
        self.condition = condition
        self.disease_type = get_param(condition, "disease_type", "chronic")  # acute, chronic, remitting
        # print(f"[DEBUG] Inside BaseNCD: {self.condition}, type: {self.disease_type}")

        self.define_pars(
            dur_condition=ss.lognorm_ex(get_param(condition, "dur_condition", 10)),
            incidence_prob=get_param(condition, "incidence", 0.01),
            incidence=ss.bernoulli(get_param(condition, "incidence", 0.01)),
            p_death=ss.bernoulli(get_param(condition, "p_death", 0.001)),
            init_prev=ss.bernoulli(get_param(condition, "init_prev", 0.05) / 100),
            remission_rate=ss.bernoulli(get_param(condition, "remmision_rate", 0.0)),
            max_disease_duration=get_param(condition, "max_disease_duration", 20),
            rel_sus=get_param(condition, "rel_sus", 1.0),
        )

        states = [
            ss.State('susceptible', default=True),  # Individuals start as susceptible
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        ]

        if self.disease_type == "remitting":
            states.append(ss.State('reversed'))
            states.append(ss.FloatArr('ti_reversed'))

        self.define_states(*states)
        self.update_pars(pars, **kwargs)

    def init_post(self):
        """ Initialize disease prevalence based on initial conditions. """
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    def step_state(self):
        if self.disease_type == "remitting":
            going_into_remission = self.pars.remission_rate.filter(self.affected.uids)
            self.affected[going_into_remission] = False
            self.reversed[going_into_remission] = True
            self.ti_reversed[going_into_remission] = self.ti

            recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
            self.reversed[recovered] = False

    def step(self):
        new_cases = self.pars.incidence.filter(self.affected.uids)
        # print(f"[DEBUG] Step {self.sim.ti}: New T2D cases = {len(new_cases)}")

        self.set_prognoses(new_cases)
        
        # Track a specific individual's `rel_sus`
        # tracked_uid = 6  # Pick an arbitrary agent
        # if tracked_uid < len(self.rel_sus):
        #     print(f"[DEBUG] Time {self.sim.ti}: Agent {tracked_uid} rel_sus = {self.rel_sus[tracked_uid]}")

        return new_cases

    def set_prognoses(self, uids):
        """ Set disease progression and mortality. """
        dur_condition = self.pars.dur_condition.rvs(uids)
        will_die = self.pars.p_death.rvs(uids)
        dead_uids = uids[will_die]

        self.affected[uids] = True
        self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt

    def update_results(self):
        """ Store prevalence for analysis. """
        super().update_results()
        prevalence = np.count_nonzero(self.affected) / len(self.sim.people)
        # print(f"[DEBUG] Time {self.ti}: {self.condition} Prevalence = {prevalence:.4f}")
        self.results.prevalence[self.ti] = prevalence


# =========================
#  BASE CLASS FOR PROGRESSIVE DISEASES (e.g., Cancer, Alzheimer's)
# =========================
class BaseDisease(ss.Disease):
    """ Base class for Progressive Diseases (e.g., Cancer, Alzheimer's). """
    def __init__(self, condition, pars=None, **kwargs):
        super().__init__()
        self.condition = condition
        # print(f"[DEBUG] Inside BaseDisease: {self.condition}")

        self.define_pars(
            dur_condition=ss.lognorm_ex(get_param(condition, "dur_condition", 10)),
            incidence_prob=get_param(condition, "incidence", 0.01),
            incidence=ss.bernoulli(get_param(condition, "incidence", 0.01)),
            p_death=ss.bernoulli(get_param(condition, "p_death", 0.001)),
            init_prev=ss.bernoulli(get_param(condition, "init_prev", 0.05) / 100),
        )

        self.define_states(
            ss.State('susceptible', default=True),  # Individuals start as susceptible
            ss.State('affected'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        )

        self.update_pars(pars, **kwargs)

    def step_state(self):
        new_cases = self.pars.incidence.filter(self.affected.uids)
        self.affected[new_cases] = True
        self.ti_affected[new_cases] = self.ti


# =========================
#  BASE CLASS FOR SIS MODEL (e.g., Flu, HPV)
# =========================
class BaseSIS(ss.SIS):
    """ Base class for Infectious Diseases (e.g., Flu, HPV). """
    def __init__(self, condition, pars=None, **kwargs):
        super().__init__()
        self.condition = condition
        # print(f"[DEBUG] Inside BaseSIS: {self.condition}")

        self.define_pars(
            dur_condition=ss.lognorm_ex(get_param(condition, "dur_condition", 10)),
            incidence_prob=get_param(condition, "incidence", 0.01),
            incidence=ss.bernoulli(get_param(condition, "incidence", 0.01)),
            init_prev=ss.bernoulli(get_param(condition, "init_prev", 0.05) / 100),
            recovery_prob=ss.bernoulli(get_param(condition, "recovery_prob", 0.1)),  # SIS needs a recovery parameter
        )

        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('infected'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
        )

        self.update_pars(pars, **kwargs)

    def step_state(self):
        new_cases = self.pars.incidence.filter(self.susceptible.uids)
        recovered_cases = self.pars.recovery_prob.filter(self.infected.uids)

        self.susceptible[recovered_cases] = True
        self.infected[recovered_cases] = False

        self.susceptible[new_cases] = False
        self.infected[new_cases] = True
        self.ti_infected[new_cases] = self.ti


# =========================
#  AUTOMATIC CLASS CREATION
# =========================
def create_disease_class(disease_name):
    """Factory function to create a disease class for a given condition."""
    model_type = get_param(disease_name, "disease_class", "ncd")  # Get model type from CSV

    if model_type == "ncd":
        base_class = BaseNCD
    elif model_type == "disease":
        base_class = BaseDisease
    elif model_type == "sis":
        base_class = BaseSIS
    else:
        raise ValueError(f"Unknown model type '{model_type}' for disease {disease_name}")

    class DiseaseClass(base_class):
        def __init__(self, pars=None, **kwargs):
            super().__init__(disease_name, pars, **kwargs)

    DiseaseClass.__name__ = disease_name.replace(" ", "").replace("-", "")  # Ensure valid class name
    return DiseaseClass




