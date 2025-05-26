import numpy as np
from scipy.stats import bernoulli, lognorm
import starsim as ss
import pandas as pd

__all__ = ['Type2Diabetes']

# Define Type2Diabetes condition
class Type2Diabetes(ss.NCD):

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        self.disease_name = "Type2Diabetes"

        disease_params = self.get_disease_parameters()

        # Calculate the mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        # Define parameters using extracted values
        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
            p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
            init_prev=ss.bernoulli(disease_params["init_prev"]),
            remission_rate=bernoulli(disease_params["remission_rate"]),  # Define remission_rate as a Bernoulli distribution
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
            affected_sex=disease_params["affected_sex"],
            p_acquire=disease_params["incidence"]
        )

        # # Define the lambda function to calculate acquisition probability
        # def calculate_p_acquire(self, sim, uids):
        #     # Start with base probability
        #     p = np.full(len(uids), self.pars.p_acquire)
            
        #     # Apply sex-specific filtering
        #     if self.pars.affected_sex == "female":
        #         # Set probability to 0 for males
        #         p[sim.people.male[uids]] = 0
        #     elif self.pars.affected_sex == "male":
        #         # Set probability to 0 for females
        #         p[sim.people.female[uids]] = 0
            
        #     # Filter out invalid indices for HIV-specific relative susceptibility
        #     valid_uids = [uid for uid in uids if uid in sim.people.hiv]
            
        #     # Apply HIV-specific relative susceptibility
        #     p[valid_uids] *= self.pars.rel_sus_hiv

        #     return p * self.rel_sus[uids]
        
        def calculate_p_acquire(self, sim, uids):
            print("DEBUG: Entered calculate_p_acquire")
            print("DEBUG: uids type:", type(uids))
            # Print a sample of uids
            try:
                print("DEBUG: uids sample:", list(uids)[:5])
            except Exception as e:
                print("DEBUG: Could not print uids sample:", e)
            p = np.full(len(uids), self.pars.p_acquire)
            if self.pars.affected_sex == "female":
                p[sim.people.male[uids]] = 0
            elif self.pars.affected_sex == "male":
                p[sim.people.female[uids]] = 0
            if hasattr(sim.people, "hiv"):
                hiv_arr = getattr(sim.people, "hiv")
                if isinstance(hiv_arr, np.ndarray):
                    hiv_positive = np.isin(uids, np.where(hiv_arr)[0])
                    p[hiv_positive] *= self.pars.rel_sus_hiv
            # Print the type and value for the first rel_sus[uid]
            try:
                first_uid = list(uids)[0]
                first_val = self.rel_sus[first_uid]
                print(f"DEBUG: first rel_sus[uid] type: {type(first_val)}, value: {first_val}")
            except Exception as e:
                print("DEBUG: Could not print rel_sus[uid] sample:", e)
            # Build rel_sus_arr robustly
            rel_sus_arr = []
            for uid in uids:
                val = self.rel_sus[uid]
                if isinstance(val, dict):
                    print(f"ERROR: rel_sus[{uid}] is a dict: {val}")
                rel_sus_arr.append(val)
            rel_sus_arr = np.array(rel_sus_arr)
            print("DEBUG: rel_sus_arr type:", type(rel_sus_arr), "dtype:", rel_sus_arr.dtype, "first 5:", rel_sus_arr[:5])
            # Try to force to float
            try:
                rel_sus_arr = rel_sus_arr.astype(float)
            except Exception as e:
                print("ERROR: Could not convert rel_sus_arr to float:", e)
                raise
            result = p * rel_sus_arr
            print("DEBUG: result type:", type(result), "dtype:", result.dtype, "first 5:", result[:5])
            return result

        self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
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

    def get_disease_parameters(self):
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()  # Remove extra spaces

        if "condition" not in df.columns:
            raise KeyError(f"Column 'condition' not found in {self.csv_path}. Available columns: {df.columns}")

        row = df[df["condition"] == self.disease_name]
        if row.empty:
            raise ValueError(f"Disease '{self.disease_name}' not found in {self.csv_path}.")

        # Extract and handle NaNs
        params = {
            "p_death": row["p_death"].values[0] if pd.notna(row["p_death"].values[0]) else 0.0001,
            "incidence": row["incidence"].values[0] if pd.notna(row["incidence"].values[0]) else 0.1,
            "dur_condition": row["dur_condition"].values[0] if pd.notna(row["dur_condition"].values[0]) else 10,  # Default 10 if missing
            "init_prev": row["init_prev"].values[0] if pd.notna(row["init_prev"].values[0]) else 0.1,
            "rel_sus_hiv": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
            "remission_rate": row["remission_rate"].values[0] if "remission_rate" in row and pd.notna(row["remission_rate"].values[0]) else 0.0,
            "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30,
            "affected_sex": row["affected_sex"].values[0] if "affected_sex" in row else "both"
        }

        return params

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
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

        return new_cases
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
#             max_disease_duration=disease_params["max_disease_duration"],
#             rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
#             affected_sex=disease_params["affected_sex"]
#         )
    
#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )
    
#         # Define the lambda function to calculate acquisition probability
#         def calculate_p_acquire(self, sim, uids):
#             # Start with base probability
#             p = np.full(len(uids), self.pars.p_acquire)
            
#             # Apply sex-specific filtering
#             if self.pars.affected_sex == "female":
#                 # Set probability to 0 for males
#                 p[sim.people.male[uids]] = 0
#             elif self.pars.affected_sex == "male":
#                 # Set probability to 0 for females
#                 p[sim.people.female[uids]] = 0
                
#             # Print base probabilities
#             # print(f"Base probabilities (p): {p}")
            
#             # Filter out invalid indices for HIV-specific relative susceptibility
#             valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
#             # Apply HIV-specific relative susceptibility
#             p[valid_uids] *= self.pars.rel_sus_hiv
#             # print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

#             return p * self.rel_sus[uids]

#         self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
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
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

#         return new_cases

# # Define Cervical Cancer condition
# class CervicalCancer(ss.NCD):
    
#     def __init__(self, disease_name='CervicalCancer', csv_path=None, pars=None, **kwargs):
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
#             max_disease_duration=disease_params["max_disease_duration"],
#             rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
#             affected_sex=disease_params["affected_sex"]  # Cervical Cancer affects only females
#         )
    
#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )
    
#         # Define the lambda function to calculate acquisition probability
#         def calculate_p_acquire(self, sim, uids):
#             # Start with base probability
#             p = np.full(len(uids), self.pars.p_acquire)
            
#             # Apply sex-specific filtering
#             if self.pars.affected_sex == "female":
#                 # Set probability to 0 for males
#                 p[sim.people.male[uids]] = 0
#             elif self.pars.affected_sex == "male":
#                 # Set probability to 0 for females
#                 p[sim.people.female[uids]] = 0
                
#             # Print base probabilities
#             # print(f"Base probabilities (p): {p}")
            
#             # Filter out invalid indices for HIV-specific relative susceptibility
#             valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
#             # Apply HIV-specific relative susceptibility
#             p[valid_uids] *= self.pars.rel_sus_hiv
#             # print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

#             return p * self.rel_sus[uids]
        
#         self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
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
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

#         return new_cases

# # Define Prostate Cancer condition
# class ProstateCancer(ss.NCD):
    
#     def __init__(self, disease_name='ProstateCancer', csv_path=None, pars=None, **kwargs):
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
#             max_disease_duration=disease_params["max_disease_duration"],
#             rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
#             affected_sex=disease_params["affected_sex"]  # Prostate Cancer affects only males
#         )
        
#         # print(f"affected_sex is: {self.pars.affected_sex}")
    
#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )
    
#         # Define the lambda function to calculate acquisition probability
#         def calculate_p_acquire(self, sim, uids):
#             # Start with base probability
#             p = np.full(len(uids), self.pars.p_acquire)
            
#             # Apply sex-specific filtering
#             if self.pars.affected_sex == "female":
#                 # Set probability to 0 for males
#                 p[sim.people.male[uids]] = 0
#             elif self.pars.affected_sex == "male":
#                 # Set probability to 0 for females
#                 p[sim.people.female[uids]] = 0
                
#             # Print base probabilities
#             # print(f"Base probabilities (p): {p}")
            
#             # Filter out invalid indices for HIV-specific relative susceptibility
#             valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
#             # Apply HIV-specific relative susceptibility
#             p[valid_uids] *= self.pars.rel_sus_hiv
#             print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

#             return p * self.rel_sus[uids]
        
#         self.p_acquire = ss.bernoulli(p=calculate_p_acquire)

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
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

#         return new_cases


# import pandas as pd
# from scipy.stats import bernoulli, lognorm
# import starsim as ss
# import numpy as np
# import sciris as sc


# class RemittingDisease(ss.NCD):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         super().__init__()
#         self.csv_path = csv_path
        
#         disease_params = self.get_disease_parameters()
        
#         # Calculate the mean in log-space (mu)
#         sigma = 0.5
#         mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

#         # Define parameters using extracted values
#         self.define_pars(
#             dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
#             incidence_prob=disease_params["incidence"],
#             p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
#             init_prev=ss.bernoulli(disease_params["init_prev"]),
#             remission_rate=bernoulli(disease_params["remission_rate"]),  # Define remission_rate as a Bernoulli distribution
#             max_disease_duration=disease_params["max_disease_duration"],
#             rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
#             affected_sex=disease_params["affected_sex"],
#             p_acquire=disease_params["incidence"]
#         )

#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )

        # # Define the lambda function to calculate acquisition probability
        # def calculate_p_acquire(self, sim, uids):
        #     # Start with base probability
        #     p = np.full(len(uids), self.pars.p_acquire)
            
        #     # Apply sex-specific filtering
        #     if self.pars.affected_sex == "female":
        #         # Set probability to 0 for males
        #         p[sim.people.male[uids]] = 0
        #     elif self.pars.affected_sex == "male":
        #         # Set probability to 0 for females
        #         p[sim.people.female[uids]] = 0
            
        #     # Filter out invalid indices for HIV-specific relative susceptibility
        #     valid_uids = [uid for uid in uids if uid in sim.people.hiv]
            
        #     # Apply HIV-specific relative susceptibility
        #     p[valid_uids] *= self.pars.rel_sus_hiv

        #     return p * self.rel_sus[uids]
        
#         self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
#         self.p_death = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_death.mean())  # Use mean to match Bernoulli
#         self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate.mean())  # Use mean to match Bernoulli

#         self.update_pars(pars, **kwargs)

#         self.define_states(
#             ss.State('susceptible', default=True),
#             ss.State('affected'),
#             ss.State('reversed'),  # New state for diabetes remission
#             ss.FloatArr('ti_affected'),
#             ss.FloatArr('ti_reversed'),
#             ss.FloatArr('ti_dead'),
#             ss.FloatArr('rel_sus', default=1.0),  # Relative susceptibility
#             ss.FloatArr('rel_death', default=1.0),  # Relative mortality
#         )
#         return

#     def get_disease_parameters(self):
#         df = pd.read_csv(self.csv_path)
#         df.columns = df.columns.str.strip()  # Remove extra spaces

#         if "condition" not in df.columns:
#             raise KeyError(f"Column 'condition' not found in {self.csv_path}. Available columns: {df.columns}")

#         row = df[df["condition"] == self.disease_name]
#         if row.empty:
#             raise ValueError(f"Disease '{self.disease_name}' not found in {self.csv_path}.")

#         # Extract and handle NaNs
#         params = {
#             "p_death": row["p_death"].values[0] if pd.notna(row["p_death"].values[0]) else 0.0001,
#             "incidence": row["incidence"].values[0] if pd.notna(row["incidence"].values[0]) else 0.1,
#             "dur_condition": row["dur_condition"].values[0] if pd.notna(row["dur_condition"].values[0]) else 10,  # Default 10 if missing
#             "init_prev": row["init_prev"].values[0] if pd.notna(row["init_prev"].values[0]) else 0.1,
#             "rel_sus_hiv": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
#             "remission_rate": row["remission_rate"].values[0] if "remission_rate" in row and pd.notna(row["remission_rate"].values[0]) else 0.0,
#             "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30,
#             "affected_sex": row["affected_sex"].values[0] if "affected_sex" in row else "both"
#         }

#         return params

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
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
#         self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

#         return new_cases

    
# # Define Chronic Disease 
# class ChronicDisease(ss.NCD):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         super().__init__()
#         self.csv_path = csv_path
        
#         disease_params = self.get_disease_parameters()
    
#         # Calculate the mean in log-space (mu)
#         sigma = 0.5
#         mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2
    
#         # Define parameters using extracted values
#         self.define_pars(
#             dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
#             incidence_prob=disease_params["incidence"],
#             p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
#             init_prev=ss.bernoulli(disease_params["init_prev"]),
#             max_disease_duration=disease_params["max_disease_duration"],
#             rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
#             affected_sex=disease_params["affected_sex"]
#         )
    
#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )
    
#         # Define the lambda function to calculate acquisition probability
#         def calculate_p_acquire(self, sim, uids):
#             # Start with base probability
#             p = np.full(len(uids), self.pars.p_acquire)
            
#             # Apply sex-specific filtering
#             if self.pars.affected_sex == "female":
#                 # Set probability to 0 for males
#                 p[sim.people.male[uids]] = 0
#             elif self.pars.affected_sex == "male":
#                 # Set probability to 0 for females
#                 p[sim.people.female[uids]] = 0
                
#             # Print base probabilities
#             # print(f"Base probabilities (p): {p}")
            
#             # Filter out invalid indices for HIV-specific relative susceptibility
#             valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
#             # Apply HIV-specific relative susceptibility
#             p[valid_uids] *= self.pars.rel_sus_hiv
#             # print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

#             return p * self.rel_sus[uids]

#         self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
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
    
#     def get_disease_parameters(self):
#         df = pd.read_csv(self.csv_path)
#         df.columns = df.columns.str.strip()  # Remove extra spaces

#         if "condition" not in df.columns:
#             raise KeyError(f"Column 'condition' not found in {self.csv_path}. Available columns: {df.columns}")

#         row = df[df["condition"] == self.disease_name]
#         if row.empty:
#             raise ValueError(f"Disease '{self.disease_name}' not found in {self.csv_path}.")

#         # Extract and handle NaNs
#         params = {
#             "p_death": row["p_death"].values[0] if pd.notna(row["p_death"].values[0]) else 0.0001,
#             "incidence": row["incidence"].values[0] if pd.notna(row["incidence"].values[0]) else 0.1,
#             "dur_condition": row["dur_condition"].values[0] if pd.notna(row["dur_condition"].values[0]) else 10,  # Default 10 if missing
#             "init_prev": row["init_prev"].values[0] if pd.notna(row["init_prev"].values[0]) else 0.1,
#             "rel_sus_hiv": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
#             "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30,
#             "affected_sex": row["affected_sex"].values[0] if "affected_sex" in row else "both"
#         }

#         return params

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
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

#         return new_cases


# class AcuteDisease(ss.NCD):
#     """ Base class for all acute diseases. """

#     def __init__(self, csv_path=None, pars=None, **kwargs):
#         super().__init__()
#         self.csv_path = csv_path
        
#         disease_params = self.get_disease_parameters()
        
#         # Calculate the mean in log-space (mu)
#         sigma = 0.5
#         mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

#         # Define parameters using extracted values
#         self.define_pars(
#             dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
#             incidence_prob=disease_params["incidence"],
#             p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
#             init_prev=ss.bernoulli(disease_params["init_prev"]),
#             max_disease_duration=disease_params["max_disease_duration"],
#             rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
#             affected_sex=disease_params["affected_sex"]
#         )

#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )

#         # Define the lambda function to calculate acquisition probability
#         def calculate_p_acquire(self, sim, uids):
#             # Start with base probability
#             p = np.full(len(uids), self.pars.p_acquire)
            
#             # Apply sex-specific filtering
#             if self.pars.affected_sex == "female":
#                 # Set probability to 0 for males
#                 p[sim.people.male[uids]] = 0
#             elif self.pars.affected_sex == "male":
#                 # Set probability to 0 for females
#                 p[sim.people.female[uids]] = 0
                
#             # Print base probabilities
#             # print(f"Base probabilities (p): {p}")
            
#             # Filter out invalid indices for HIV-specific relative susceptibility
#             valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
#             # Apply HIV-specific relative susceptibility
#             p[valid_uids] *= self.pars.rel_sus_hiv
#             # print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

#             return p * self.rel_sus[uids]

#         self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
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

#     def get_disease_parameters(self):
#         df = pd.read_csv(self.csv_path)
#         df.columns = df.columns.str.strip()  # Remove extra spaces

#         if "condition" not in df.columns:
#             raise KeyError(f"Column 'condition' not found in {self.csv_path}. Available columns: {df.columns}")

#         row = df[df["condition"] == self.disease_name]
#         if row.empty:
#             raise ValueError(f"Disease '{self.disease_name}' not found in {self.csv_path}.")

#         # Extract and handle NaNs
#         params = {
#             "p_death": row["p_death"].values[0] if pd.notna(row["p_death"].values[0]) else 0.0001,
#             "incidence": row["incidence"].values[0] if pd.notna(row["incidence"].values[0]) else 0.1,
#             "dur_condition": row["dur_condition"].values[0] if pd.notna(row["dur_condition"].values[0]) else 10,  # Default 10 if missing
#             "init_prev": row["init_prev"].values[0] if pd.notna(row["init_prev"].values[0]) else 0.1,
#             "rel_sus_hiv": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
#             "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30,
#             "affected_sex": row["affected_sex"].values[0] if "affected_sex" in row else "both"
#         }

#         return params

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
#         self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

#         return new_cases


# class GenericSIS(ss.SIS):
#     """ Base class for communicable diseases using the SIS model. """

#     def __init__(self, csv_path, pars=None, **kwargs):
#         super().__init__()
#         self.csv_path = csv_path
#         disease_params = self.get_disease_parameters()
        
#         # Define parameters using extracted values
#         sigma = 0.5
#         mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

#         # Define parameters using extracted values
#         self.define_pars(
#             dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
#             incidence_prob=disease_params["incidence"],
#             p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
#             init_prev=ss.bernoulli(disease_params["init_prev"]),
#             max_disease_duration=disease_params["max_disease_duration"],
#             rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
#             affected_sex=disease_params["affected_sex"],
#             p_acquire=disease_params["incidence"]
#         )

#         # Define disease parameters
#         self.define_pars(
#             p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
#         )

#         # Define the lambda function to calculate acquisition probability
#         def calculate_p_acquire(self, sim, uids):
#             # Start with base probability
#             p = np.full(len(uids), self.pars.p_acquire)
            
#             # Apply sex-specific filtering
#             if self.pars.affected_sex == "female":
#                 # Set probability to 0 for males
#                 p[sim.people.male[uids]] = 0
#             elif self.pars.affected_sex == "male":
#                 # Set probability to 0 for females
#                 p[sim.people.female[uids]] = 0
            
#             # Filter out invalid indices for HIV-specific relative susceptibility
#             valid_uids = [uid for uid in uids if uid in sim.people.hiv]
            
#             # Apply HIV-specific relative susceptibility
#             p[valid_uids] *= self.pars.rel_sus_hiv

#             return p * self.rel_sus[uids]
        
#         self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
#         self.p_death = ss.bernoulli(p=lambda self, sim, uids: self.pars.p_death.mean())  # Use mean to match Bernoulli
#         self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate.mean())  # Use mean to match Bernoulli

#         self.update_pars(pars, **kwargs)

#         self.define_states(
#             ss.State('susceptible', default=True),
#             ss.State('infected'),
#             ss.FloatArr('ti_infected'),
#             ss.FloatArr('ti_reversed'),
#             ss.FloatArr('ti_dead'),
#             ss.FloatArr('rel_sus', default=1.0),  # Relative susceptibility
#             ss.FloatArr('rel_death', default=1.0),  # Relative mortality
#         )
#         return
    
#     def get_disease_parameters(self):
#         df = pd.read_csv(self.csv_path)
#         df.columns = df.columns.str.strip()  # Remove extra spaces

#         if "condition" not in df.columns:
#             raise KeyError(f"Column 'condition' not found in {self.csv_path}. Available columns: {df.columns}")

#         row = df[df["condition"] == self.disease_name]
#         if row.empty:
#             raise ValueError(f"Disease '{self.disease_name}' not found in {self.csv_path}.")

#         # Extract and handle NaNs
#         params = {
#             "p_death": row["p_death"].values[0] if pd.notna(row["p_death"].values[0]) else 0.0001,
#             "incidence": row["incidence"].values[0] if pd.notna(row["incidence"].values[0]) else 0.1,
#             "dur_condition": row["dur_condition"].values[0] if pd.notna(row["dur_condition"].values[0]) else 10,  # Default 10 if missing
#             "init_prev": row["init_prev"].values[0] if pd.notna(row["init_prev"].values[0]) else 0.1,
#             "rel_sus_hiv": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
#             "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30,
#             "affected_sex": row["affected_sex"].values[0] if "affected_sex" in row else "both"
#         }

#         return params

#     def init_post(self):
#         initial_cases = self.pars.init_prev.filter()
#         self.set_prognoses(initial_cases)
#         return initial_cases

#     def set_prognoses(self, uids):
#         sim = self.sim
#         p = self.pars
#         self.susceptible[uids] = False
#         self.infected[uids] = True
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
#         self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)
#         return

#     def step(self):
#         ti = self.ti

#         # New cases
#         susceptible = (~self.infected).uids
#         new_cases = self.p_acquire.filter(susceptible)
#         self.infected[new_cases] = True
#         self.ti_infected[new_cases] = ti

#         # Death
#         deaths = self.p_death.filter(new_cases)  # Applied only to people just affected
#         self.sim.people.request_death(deaths)
#         self.ti_dead[deaths] = ti

#         # Results
#         self.results.new_cases[ti] = len(new_cases)
#         self.results.new_deaths[ti] = len(deaths)
#         self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

#         return new_cases              
        
# class Type1Diabetes(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Type1Diabetes'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Type1Diabetes'
#         )
#         return

# class Type2Diabetes(RemittingDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Type2Diabetes'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Type2Diabetes'
#         )
#         return

# class Hypertension(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Hypertension'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Hypertension'
#         )
#         return

# class Obesity(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Obesity'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Obesity'
#         )
#         return

# class CardiovascularDiseases(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'CardiovascularDiseases'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'CardiovascularDiseases'
#         )
#         return

# class ChronicKidneyDisease(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'ChronicKidneyDisease'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'ChronicKidneyDisease'
#         )
#         return

# class Hyperlipidemia(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Hyperlipidemia'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Hyperlipidemia'
#         )
#         return

# class CervicalCancer(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'CervicalCancer'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'CervicalCancer'
#         )
#         return

# class ColorectalCancer(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'ColorectalCancer'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'ColorectalCancer'
#         )
#         return

# class BreastCancer(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'BreastCancer'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'BreastCancer'
#         )
#         return

# class LungCancer(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'LungCancer'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'LungCancer'
#         )
#         return

# class ProstateCancer(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'ProstateCancer'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'ProstateCancer'
#         )
#         return

# class AlcoholUseDisorder(RemittingDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'AlcoholUseDisorder'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'AlcoholUseDisorder'
#         )
#         return

# class TobaccoUse(RemittingDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'TobaccoUse'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'TobaccoUse'
#         )
#         return

# class HIVAssociatedDementia(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'HIVAssociatedDementia'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'HIVAssociatedDementia'
#         )
#         return

# class PTSD(RemittingDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'PTSD'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'PTSD'
#         )
#         return

# class Depression(RemittingDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Depression'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Depression'
#         )
#         return

# class DomesticViolence(AcuteDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'DomesticViolence'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'DomesticViolence'
#         )
#         return

# class RoadInjuries(AcuteDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'RoadInjuries'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'RoadInjuries'
#         )
#         return

# class ChronicLiverDisease(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'ChronicLiverDisease'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'ChronicLiverDisease'
#         )
#         return

# class Asthma(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Asthma'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Asthma'
#         )
#         return

# class COPD(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'COPD'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'COPD'
#         )
#         return

# class AlzheimersDisease(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'AlzheimersDisease'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'AlzheimersDisease'
#         )
#         return

# class ParkinsonsDisease(ChronicDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'ParkinsonsDisease'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'ParkinsonsDisease'
#         )
#         return

# class ViralHepatitis(AcuteDisease):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'ViralHepatitis'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'ViralHepatitis'
#         )
#         return

# class HPV(GenericSIS):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'HPV'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'HPV'
#         )
#         return

# class Flu(GenericSIS):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'Flu'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'Flu'
#         )
#         return

# class TB(GenericSIS):
#     def __init__(self, csv_path, pars=None, **kwargs):
#         self.disease_name = 'TB'
#         super().__init__(csv_path, pars, **kwargs)
#         self.define_pars(
#             label = 'TB'
#         )
#         return
    