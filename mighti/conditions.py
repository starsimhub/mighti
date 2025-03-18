import numpy as np
from scipy.stats import bernoulli, lognorm
import starsim as ss
import pandas as pd

__all__ = ['Type2Diabetes', 'ChronicKidneyDisease']

def get_disease_parameters(disease_name, csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces

    if "condition" not in df.columns:
        raise KeyError(f"Column 'condition' not found in {csv_path}. Available columns: {df.columns}")

    row = df[df["condition"] == disease_name]
    if row.empty:
        raise ValueError(f"Disease '{disease_name}' not found in {csv_path}.")

    # Extract and handle NaNs
    params = {
        "p_death": row["p_death"].values[0] if pd.notna(row["p_death"].values[0]) else 0.0001,
        "incidence": row["incidence"].values[0] if pd.notna(row["incidence"].values[0]) else 0.1,
        "dur_condition": row["dur_condition"].values[0] if pd.notna(row["dur_condition"].values[0]) else 10,  # Default 10 if missing
        "init_prev": row["init_prev"].values[0] if pd.notna(row["init_prev"].values[0]) else 0.1,
        "rel_sus": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
        "remission_rate": row["remmision_rate"].values[0] if pd.notna(row["remmision_rate"].values[0]) else 0.0,
        "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30
    }

    return params


# Define Type2Diabetes condition
class Type2Diabetes(ss.NCD):
    
    def __init__(self, disease_name='Type2Diabetes', csv_path=None, pars=None, **kwargs):
            super().__init__()
            self.disease_name = disease_name
            self.csv_path = csv_path
            disease_params = get_disease_parameters(disease_name, csv_path)
    
            # Calculate the mean in log-space (mu)
            sigma = 0.5
            mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2
    
            # Define parameters using extracted values
            self.define_pars(
                dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
                incidence_prob=disease_params["incidence"],
                p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
                init_prev=ss.bernoulli(disease_params["init_prev"]),
                remission_rate=bernoulli(disease_params["remission_rate"]),  # Define remission_rate as a Bernoulli distribution
                max_disease_duration=disease_params["max_disease_duration"]
            )
    
            # Define disease parameters
            self.define_pars(
                p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
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
    
    def __init__(self, disease_name='ChronicKidneyDisease', csv_path=None, pars=None, **kwargs):
        super().__init__()
        self.disease_name = disease_name
        self.csv_path = csv_path
        disease_params = get_disease_parameters(disease_name, csv_path)
    
        # Calculate the mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2
    
        # Define parameters using extracted values
        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
            incidence_prob=disease_params["incidence"],
            p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
            init_prev=ss.bernoulli(disease_params["init_prev"]),
            max_disease_duration=disease_params["max_disease_duration"]
        )
    
        # Define disease parameters
        self.define_pars(
            p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
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

# import numpy as np
# from scipy.stats import bernoulli, lognorm
# import starsim as ss
# import pandas as pd

# # CONDITIONS
# # This is an umbrella term for any health condition. Some conditions can lead directly
# # to death/disutility (e.g. heart disease, HIV, depression), while others do not. All
# # conditions can affect the (1) risk of acquiring, (2) persistence of, (3) severity of
# # other conditions.

# # Global variables to store disease lists
# ncds = []  # Ensures ncds exists even if initialize_conditions() fails
# communicable_diseases = []

# def get_disease_parameters(disease_name, csv_path):
#     df = pd.read_csv(csv_path)
#     df.columns = df.columns.str.strip()  # Remove extra spaces

#     if "condition" not in df.columns:
#         raise KeyError(f"Column 'condition' not found in {csv_path}. Available columns: {df.columns}")

#     row = df[df["condition"] == disease_name]
#     if row.empty:
#         raise ValueError(f"Disease '{disease_name}' not found in {csv_path}.")

#     # Extract and handle NaNs
#     params = {
#         "p_death": row["p_death"].values[0] if pd.notna(row["p_death"].values[0]) else 0.0001,
#         "incidence": row["incidence"].values[0] if pd.notna(row["incidence"].values[0]) else 0.1,
#         "dur_condition": row["dur_condition"].values[0] if pd.notna(row["dur_condition"].values[0]) else 10,  # Default 10 if missing
#         "init_prev": row["init_prev"].values[0] if pd.notna(row["init_prev"].values[0]) else 0.1,
#         "rel_sus": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
#         "remission_rate": row["remmision_rate"].values[0] if pd.notna(row["remmision_rate"].values[0]) else 0.0,
#         "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30,
#         "affected_sex": row["affected_sex"].values[0] if "affected_sex" in row else "both"
#     }
    
#     return params



# # Define Type2Diabetes condition
# class Type2Diabetes(ss.NCD):
    
#     def __init__(self, disease_name='Type2Diabetes', csv_path=None, pars=None, **kwargs):
#             super().__init__()
#             self.disease_name = disease_name
#             self.csv_path = csv_path
#             disease_params = get_disease_parameters(disease_name, csv_path)
    
#             # Calculate the mean in log-space (mu)
#             sigma = 0.5
#             mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2
    
#             # Define parameters using extracted values
#             self.define_pars(
#                 dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
#                 incidence_prob=disease_params["incidence"],
#                 p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
#                 init_prev=ss.bernoulli(disease_params["init_prev"]),
#                 remission_rate=bernoulli(disease_params["remission_rate"]),  # Define remission_rate as a Bernoulli distribution
#                 max_disease_duration=disease_params["max_disease_duration"]
#             )
    
#             # Define disease parameters
#             self.define_pars(
#                 p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#             )
    
#             self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_acquire * self.rel_sus[uids])
#             self.p_death = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_death.mean())  # Use mean to match Bernoulli
#             self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate.mean())  # Use mean to match Bernoulli
    
#             self.update_pars(pars, **kwargs)
    
#             self.define_states(
#                 ss.State('susceptible', default=True),
#                 ss.State('affected'),
#                 ss.State('reversed'),  # New state for diabetes remission
#                 ss.FloatArr('ti_affected'),
#                 ss.FloatArr('ti_reversed'),
#                 ss.FloatArr('ti_dead'),
#                 ss.FloatArr('rel_sus', default=1.0),  # Relative susceptibility
#                 ss.FloatArr('rel_death', default=1.0),  # Relative mortality
#             )
#             return

#     def init_post(self):
#         initial_cases = self.pars.init_prev.filter()
#         self.set_prognoses(initial_cases)
#         return initial_cases

#     def set_prognoses(self, uids):
#         sim = self.sim
#         p = self.pars
#         self.susceptible[uids] = False
#         self.affected[uids] = True
#         dur_condition = p.dur_condition.rvs(size=len(uids))
#         will_die = p.p_death.rvs(size=len(uids))
#         dead_uids = uids[will_die]
#         rec_uids = uids[~will_die]
#         self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt
#         self.ti_reversed[rec_uids] = self.ti + dur_condition[~will_die] / self.t.dt
#         return

#     def init_results(self):
#         super().init_results()
#         existing_results = set(self.results.keys())
        
#         if 'new_cases' not in existing_results:
#             self.define_results(ss.Result('new_cases', dtype=int, label='New Cases'))
#         if 'new_deaths' not in existing_results:
#             self.define_results(ss.Result('new_deaths', dtype=int, label='Deaths'))
#         if 'prevalence' not in existing_results:
#             self.define_results(ss.Result('prevalence', dtype=float, label='Prevalence'))
#         if 'remission_prevalence' not in existing_results:
#             self.define_results(ss.Result('remission_prevalence', dtype=float, label='Remission Prevalence'))
#         if 'reversal_prevalence' not in existing_results:
#             self.define_results(ss.Result('reversal_prevalence', dtype=float))
        
#         return

#     def update_results(self):
#         super().update_results()
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
#         self.results.reversal_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
#         self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
#         return

#     def step_state(self):
#         # Handle remission (reversal)
#         going_into_remission = self.p_remission.filter(self.affected.uids)  # Use p_remission for filtering
#         self.affected[going_into_remission] = False
#         self.reversed[going_into_remission] = True
#         self.ti_reversed[going_into_remission] = self.ti

#         # Handle recovery, death, and beta-cell function exhaustion
#         recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
#         self.reversed[recovered] = False
#         self.susceptible[recovered] = True  # Recovered individuals become susceptible again
#         deaths = (self.ti_dead == self.ti).uids
#         self.sim.people.request_death(deaths)
#         self.results.new_deaths[self.ti] = len(deaths)

#     def step(self):
#         ti = self.ti

#         # New cases
#         susceptible = (~self.affected).uids
#         new_cases = self.p_acquire.filter(susceptible)
#         self.affected[new_cases] = True
#         self.ti_affected[new_cases] = ti

#         # Death
#         deaths = self.p_death.filter(new_cases)  # Applied only to people just affected
#         self.sim.people.request_death(deaths)
#         self.ti_dead[deaths] = ti

#         # Results
#         self.results.new_cases[ti] = len(new_cases)
#         self.results.new_deaths[ti] = len(deaths)
#         self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(self.sim.people)
#         self.results.remission_prevalence[ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

#         return new_cases


# # Define Chronic Kidney Disease condition
# class ChronicKidneyDisease(ss.NCD):
    
#     def __init__(self, disease_name='ChronicKidneyDisease', csv_path=None, pars=None, **kwargs):
#         super().__init__()
#         self.disease_name = disease_name
#         self.csv_path = csv_path
#         disease_params = get_disease_parameters(disease_name, csv_path)
    
#         # Calculate the mean in log-space (mu)
#         sigma = 0.5
#         mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2
    
#         # Define parameters using extracted values
#         self.define_pars(
#             dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
#             incidence_prob=disease_params["incidence"],
#             p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
#             init_prev=ss.bernoulli(disease_params["init_prev"]),
#             max_disease_duration=disease_params["max_disease_duration"]
#         )
    
#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )
    
#         self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_acquire * self.rel_sus[uids])
#         self.p_death = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_death.mean())  # Use mean to match Bernoulli
    
#         self.update_pars(pars, **kwargs)
    
#         self.define_states(
#             ss.State('susceptible', default=True),
#             ss.State('affected'),
#             ss.FloatArr('ti_affected'),
#             ss.FloatArr('ti_dead'),
#             ss.FloatArr('rel_sus', default=1.0),  # Relative susceptibility
#             ss.FloatArr('rel_death', default=1.0),  # Relative mortality
#         )
#         return

#     def init_post(self):
#         initial_cases = self.pars.init_prev.filter()
#         self.set_prognoses(initial_cases)
#         return initial_cases

#     def set_prognoses(self, uids):
#         sim = self.sim
#         p = self.pars
#         self.susceptible[uids] = False
#         self.affected[uids] = True
#         dur_condition = p.dur_condition.rvs(size=len(uids))
#         will_die = p.p_death.rvs(size=len(uids))
#         dead_uids = uids[will_die]
#         self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt
#         return

#     def init_results(self):
#         super().init_results()
#         existing_results = set(self.results.keys())
        
#         if 'new_cases' not in existing_results:
#             self.define_results(ss.Result('new_cases', dtype=int, label='New Cases'))
#         if 'new_deaths' not in existing_results:
#             self.define_results(ss.Result('new_deaths', dtype=int, label='Deaths'))
#         if 'prevalence' not in existing_results:
#             self.define_results(ss.Result('prevalence', dtype=float, label='Prevalence'))
        
#         return

#     def update_results(self):
#         super().update_results()
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
#         return

#     def step(self):
#         ti = self.ti

#         # New cases
#         susceptible = (~self.affected).uids
#         new_cases = self.p_acquire.filter(susceptible)
#         self.affected[new_cases] = True
#         self.ti_affected[new_cases] = ti

#         # Death
#         deaths = self.p_death.filter(new_cases)  # Applied only to people just affected
#         self.sim.people.request_death(deaths)
#         self.ti_dead[deaths] = ti

#         # Results
#         self.results.new_cases[ti] = len(new_cases)
#         self.results.new_deaths[ti] = len(deaths)
#         self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(self.sim.people)

#         return new_cases    
    
# # def initialize_conditions(df, ncd_list, sis_list):
# #     """ Load disease parameters and categories from `mighti_main.py`. """
# #     global df_params, ncds, communicable_diseases
# #     df_params = df  # Store the entire DataFrame globally
# #     ncds = ncd_list
# #     communicable_diseases = sis_list
# #     # Register disease classes right after receiving the parameters
# #     register_disease_classes()
    

# # def register_disease_classes():
# #     """ Dynamically create and register disease classes in `mighti`. """
# #     global disease_classes
# #     disease_classes = {}  # Reset to avoid duplicates

# #     # Create NCD classes dynamically
# #     for disease in ncds:
# #         class_definition = type(
# #             disease,  # Class name (e.g., "Type2Diabetes")
# #             (GenericNCD,),  # Base class (NCD)
# #             {
# #                 "__init__": (lambda disease_name: 
# #                     lambda self, pars=None, **kwargs: 
# #                     GenericNCD.__init__(self, disease_name, pars, **kwargs)
# #                 )(disease)  # Correct lambda closure
# #             }
# #         )
# #         globals()[disease] = class_definition  # Add to global namespace
# #         disease_classes[disease] = class_definition  # Store in dictionary

# #     # Create SIS (communicable disease) classes dynamically
# #     for disease in communicable_diseases:
# #         class_definition = type(
# #             disease,  # Class name (e.g., "Flu")
# #             (GenericSIS,),  # Base class for communicable diseases
# #             {
# #                 "__init__": (lambda disease_name: 
# #                     lambda self, pars=None, **kwargs: 
# #                     GenericSIS.__init__(self, disease_name, pars, **kwargs)
# #                 )(disease)  # Correct lambda closure
# #             }
# #         )
# #         globals()[disease] = class_definition  # Add to global namespace
# #         disease_classes[disease] = class_definition  # Store in dictionary

# #     # Register dynamically created classes in `mighti`
# #     import mighti
# #     mighti.__dict__.update(disease_classes)
    
    


# # class GenericNCD(ss.NCD):
# #     """ Base class for all Non-Communicable Diseases (NCDs). """

# #     def __init__(self, disease_name, csv_path, pars=None, **kwargs):
# #         super().__init__()
# #         self.disease_name = disease_name
# #         disease_params = get_disease_parameters(disease_name)

# #         # Define parameters using extracted values
# #         self.define_pars(
# #         dur_condition=ss.lognorm_ex(disease_params["dur_condition"] or 10),  # Default = 10
# #         incidence_prob=disease_params["incidence_prob"],
# #         incidence=ss.bernoulli(disease_params["incidence_prob"]),
# #         p_death=ss.bernoulli(disease_params["p_death"]),
# #         init_prev=ss.bernoulli(disease_params["init_prev"]),
# #         remission_rate=ss.bernoulli(disease_params["remission_rate"]),
# #         max_disease_duration=disease_params["max_disease_duration"] or 30  # Default = 30
# #     )

# #         # Define base states
# #         states = [
# #             ss.State('affected'),
# #             ss.FloatArr('ti_affected'),
# #             ss.FloatArr('ti_dead'),
# #             ss.FloatArr('rel_sus', default=1.0),
# #         ]
        
# #         # Only add "susceptible" for diseases that are not lifelong
# #         if disease_params["incidence_prob"] > 0:
# #             states.insert(0, ss.State('susceptible', default=True))

# #         # Add remission states only for remitting diseases
# #         if disease_params["remission_rate"]:
# #             states.append(ss.State('reversed'))
# #             states.append(ss.FloatArr('ti_reversed'))

# #         self.define_states(*states)
# #         self.update_pars(pars, **kwargs)
    
# #     def init_post(self):
# #         """ Initialize disease prevalence based on `init_prev`. """
# #         initial_cases = self.pars.init_prev.filter()
# #         # print(f"{self.disease_name}: Expected initial cases: {len(initial_cases)}")
    
# #         if len(initial_cases) == 0:
# #             print(f"WARNING: {self.disease_name}: `init_prev` is filtering 0 cases! Something is wrong.")
    
# #         # Debug: Check the initial state of rel_sus
# #         # print(f"[DEBUG] {self.disease_name} → rel_sus at init_post(): {self.rel_sus}")
    
# #         self.set_prognoses(initial_cases)
    
# #         # Debug: Check rel_sus after set_prognoses
# #         # print(f"[DEBUG] {self.disease_name} → rel_sus after set_prognoses(): {self.rel_sus}")
    
# #         return initial_cases
    
        
# #     def step(self):
# #         """ Process new disease cases based on incidence. """
        
# #         # Print basic debugging info
# #         # print(f"[DEBUG] {self.disease_name} → rel_sus at start of step(): mean={self.rel_sus.mean()}, min={self.rel_sus.min()}, max={self.rel_sus.max()}")
        
# #         if hasattr(self, "susceptible"):  
# #             new_cases = self.pars.incidence.filter(self.susceptible.uids)
# #             self.set_prognoses(new_cases)
    
# #         # Print after running new cases
# #         # print(f"[DEBUG] {self.disease_name} → rel_sus at end of step(): mean={self.rel_sus.mean()}, min={self.rel_sus.min()}, max={self.rel_sus.max()}")    
        

# #     def step_state(self):
# #         """ Handle remission, recovery, and deaths. """
# #         # Process remission (for remitting diseases)
# #         if hasattr(self, "reversed"):
# #             going_into_remission = self.pars.remission_rate.filter(self.affected.uids)
# #             self.affected[going_into_remission] = False
# #             self.reversed[going_into_remission] = True
# #             self.ti_reversed[going_into_remission] = self.ti

# #             # Recover from remission (return to susceptible)
# #             recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
# #             self.reversed[recovered] = False
# #             if hasattr(self, "susceptible"):
# #                 self.susceptible[recovered] = True  

# #         # Handle deaths
# #         deaths = (self.ti_dead == self.ti).uids
# #         self.sim.people.request_death(deaths)

        
# #     def set_prognoses(self, uids):
# #         """ Assign disease progression, including deaths and remission. """
# #         if len(uids) == 0:
# #             print(f"WARNING: {self.disease_name}: No affected cases! This may be an issue.")
    
# #         # print(f"[DEBUG] {self.disease_name} → rel_sus before updating in set_prognoses(): {self.rel_sus}")
    
# #         self.affected[uids] = True  
# #         if hasattr(self, "susceptible"):
# #             self.susceptible[uids] = False 
    
# #         dur_condition = self.pars.dur_condition.rvs(uids)
# #         will_die = self.pars.p_death.rvs(uids)
# #         dead_uids = uids[will_die]
# #         rec_uids = uids[~will_die]
    
# #         if hasattr(self, "reversed"):
# #             self.ti_reversed[rec_uids] = self.ti + dur_condition[~will_die] / self.t.dt
# #             self.reversed[rec_uids] = True
# #         self.ti_dead[dead_uids] = self.ti + dur_condition[will_die] / self.t.dt
    
# #         # Debug: Check rel_sus after set_prognoses
# #         # print(f"[DEBUG] {self.disease_name} → rel_sus after updating in set_prognoses(): {self.rel_sus}")
        

# #     def init_results(self):
# #         """ Define results tracking for prevalence and remission. """
# #         super().init_results()
# #         self.define_results(
# #             ss.Result('reversal_prevalence', dtype=float),
# #         )
# #         return

# #     def update_results(self):
# #         """ Store prevalence for analysis. """
# #         super().update_results()

# #         total_population = len(self.sim.people)
# #         if total_population == 0:
# #             return  

# #         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / total_population

# #         if hasattr(self, "reversed"):
# #             self.results.reversal_prevalence[self.ti] = np.count_nonzero(self.reversed) / total_population


# # class GenericSIS(ss.SIS):
# #     """ Base class for communicable diseases using the SIS model. """

# #     def __init__(self, disease_name, csv_path, pars=None, **kwargs):
# #         super().__init__()
# #         self.disease_name = disease_name

# #         # Fetch disease parameters from the globally stored DataFrame
# #         disease_params = get_disease_parameters(disease_name)

# #         # Define parameters using extracted values
# #         self.define_pars(
# #             dur_condition=ss.lognorm_ex(disease_params["dur_condition"]),
# #             incidence_prob=disease_params["incidence_prob"],
# #             incidence=ss.bernoulli(disease_params["incidence_prob"]),
# #             p_death=ss.bernoulli(disease_params["p_death"]),
# #             init_prev=ss.bernoulli(disease_params["init_prev"]),
# #             recovery_prob=ss.bernoulli(disease_params.get("recovery_prob", 0.1)),  # Use CSV if available
# #         )

# #         self.define_states(
# #             ss.State('susceptible', default=True),
# #             ss.State('infected'),
# #             ss.FloatArr('ti_infected'),
# #             ss.FloatArr('ti_dead'),
# #         )

# #         self.update_pars(pars, **kwargs)

# #     def step_state(self):
# #         """Handles transitions between states in an SIS model (new infections & recoveries)."""
# #         new_cases = self.pars.incidence.filter(self.susceptible.uids)
# #         recovered_cases = self.pars.recovery_prob.filter(self.infected.uids)

# #         # Process recoveries
# #         self.susceptible[recovered_cases] = True
# #         self.infected[recovered_cases] = False

# #         # Process new infections
# #         self.susceptible[new_cases] = False
# #         self.infected[new_cases] = True
# #         self.ti_infected[new_cases] = self.ti  # Track infection time

        
        
# # # Dictionary to store dynamically created disease classes
# # disease_classes = {}

# # # Create NCD classes dynamically
# # for disease in ncds:
# #     disease_class = type(
# #         disease,  # Class name (e.g., "Type2Diabetes")
# #         (GenericNCD,),  # Base class (NCD)
# #         {"__init__": lambda self, pars=None, disease=disease, **kwargs: 
# #             super(self.__class__, self).__init__(disease, pars, **kwargs)}
# #     )
# #     globals()[disease] = disease_class  # Add to global namespace
# #     disease_classes[disease] = disease_class  # Store in dictionary

# # # Create SIS (communicable disease) classes dynamically
# # for disease in communicable_diseases:
# #     disease_class = type(
# #         disease,  # Class name (e.g., "Flu")
# #         (GenericSIS,),  # Base class for communicable diseases (to be defined)
# #         {"__init__": lambda self, pars=None, disease=disease, **kwargs: 
# #             super(self.__class__, self).__init__(disease, pars, **kwargs)}
# #     )
# #     globals()[disease] = disease_class  # Add to global namespace
# #     disease_classes[disease] = disease_class  # Store in dictionary
    