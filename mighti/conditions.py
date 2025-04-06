import numpy as np
from scipy.stats import bernoulli, lognorm
import starsim as ss
import pandas as pd

__all__ = ['Type2Diabetes', 'ChronicKidneyDisease', 'CervicalCancer','ProstateCancer']

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
        "rel_sus_hiv": row["rel_sus"].values[0] if pd.notna(row["rel_sus"].values[0]) else 1.0,
        "remission_rate": row["remission_rate"].values[0] if "remission_rate" in row and pd.notna(row["remission_rate"].values[0]) else 0.0,
        "max_disease_duration": row["max_disease_duration"].values[0] if pd.notna(row["max_disease_duration"].values[0]) else 30,
        "affected_sex": row["affected_sex"].values[0] if "affected_sex" in row else "both"
    }

    return params

    
def print_age_distribution(sim, prevalence_data, age_bins):
    ages = sim.people.age
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_distribution, _ = np.histogram(ages, bins)
    # print("Age Distribution:")
    # for i in range(len(bins) - 1):
        # print(f"{bins[i]}-{bins[i+1]}: {age_distribution[i]}")

    # # Debug: Print the structure of prevalence_data
    # print("Prevalence Data Structure:")
    # for disease, data in prevalence_data.items():
    #     print(f"Disease: {disease}")
    #     for sex, age_data in data.items():
    #         print(f"  Sex: {sex}")
    #         for age, value in age_data.items():
    #             print(f"    Age: {age}, Value: {value}")

    # Calculate mean prevalence for each age group
    prevalence_means = []
    for i in range(len(bins) - 1):
        age_group_indices = (ages >= bins[i]) & (ages < bins[i+1])
        age_group_ages = ages[age_group_indices]
        if len(age_group_ages) == 0:
            prevalence_means.append(0)
            continue

        sex_group = sim.people.female[age_group_indices]
        prevalence_values = []
        for j, age in enumerate(age_group_ages):
            sex = 'female' if sex_group[j] else 'male'
            for k in range(len(age_bins) - 1):
                left = age_bins[k]
                right = age_bins[k + 1]
                if age >= left and age < right:
                    prevalence_values.append(prevalence_data['Type2Diabetes'][sex][left])
                    break
            if age >= age_bins[-1]:  # For ages at or above highest bin
                prevalence_values.append(prevalence_data['Type2Diabetes'][sex][age_bins[-1]])

        mean_prevalence = np.mean(prevalence_values) if prevalence_values else 0
        prevalence_means.append(mean_prevalence)

    # print("Mean Prevalence for each Age Group:")
    # for i in range(len(bins) - 1):
    #     print(f"{bins[i]}-{bins[i+1]}: {prevalence_means[i]:.6f}")

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
            param_dict = dict(
                dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
                incidence_prob=disease_params["incidence"],
                p_death=bernoulli(disease_params["p_death"]),  # Define p_death as a Bernoulli distribution
                init_prev=ss.bernoulli(disease_params["init_prev"]),
                remission_rate=bernoulli(disease_params["remission_rate"]),  # Define remission_rate as a Bernoulli distribution
                max_disease_duration=disease_params["max_disease_duration"],
                rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
                affected_sex=disease_params["affected_sex"],
                p_acquire=disease_params["incidence"]

            )
    

        # Check if init_prev was provided in pars
            if pars is not None and 'init_prev' in pars:
                # Use the age-sex dependent prevalence from pars
                # print(f"Using age-sex dependent prevalence for {disease_name}")
                param_dict['init_prev'] = pars['init_prev']
            else:
                # Fallback to static value from CSV
                # print(f"Using static prevalence for {disease_name} from CSV: {disease_params['init_prev']}")
                param_dict['init_prev'] = ss.bernoulli(disease_params["init_prev"])
            
            # Define all parameters at once
            self.define_pars(**param_dict)
            
            # Process any remaining parameters not handled above
            remaining_pars = {} if pars is None else {k: v for k, v in pars.items() if k != 'init_prev'}
            self.update_pars(remaining_pars, **kwargs)
                
    
            # Define the lambda function to calculate acquisition probability
            def calculate_p_acquire(self, sim, uids):
                # Start with base probability
                p = np.full(len(uids), self.pars.p_acquire)
                
                # Apply sex-specific filtering
                if self.pars.affected_sex == "female":
                    # Set probability to 0 for males
                    p[sim.people.male[uids]] = 0
                elif self.pars.affected_sex == "male":
                    # Set probability to 0 for females
                    p[sim.people.female[uids]] = 0
            
                # Print base probabilities
                # print(f"Base probabilities (p): {p}")
                
                # Filter out invalid indices for HIV-specific relative susceptibility
                valid_uids = [uid for uid in uids if uid in sim.people.hiv]
            
                # Apply HIV-specific relative susceptibility
                p[valid_uids] *= self.pars.rel_sus_hiv
                # print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

                return p * self.rel_sus[uids]
            
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

    def init_post(self):
        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)
        return initial_cases

    
    # def init_post(self):
    #     print("\n=== CUSTOM TYPE2DIABETES INITIALIZATION ===")
    #     sim = self.sim
        
    #     # Ensure age_bins is available
    #     age_bins = getattr(self.pars.init_prev.pars, 'age_bins', None)
    #     if age_bins is None:
    #         # Provide default age bins if not available
    #         age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #         print("Warning: age_bins not found in parameters. Using default age bins.")
        
    #     # Print age distribution and mean prevalence
    #     print_age_distribution(sim, self.pars.init_prev.pars.p, age_bins)
        
    #     # Get all people
    #     n_people = len(sim.people)
    #     all_uids = np.arange(n_people)
        
    #     # Get the prevalence function
    #     init_func = self.pars.init_prev.pars.p
        
    #     # Convert to UIDs for the function call
    #     uids_obj = ss.uids(all_uids)
        
    #     # Calculate probabilities
    #     probs = init_func(self, sim, uids_obj)
    #     mean_prob = np.mean(probs)
    #     print(f"Probability stats: mean={mean_prob:.6f}, min={np.min(probs):.6f}, max={np.max(probs):.6f}")
        
    #     # Expected cases based on mean probability
    #     expected_cases = int(mean_prob * n_people)
    #     print(f"Expected number of cases based on probabilities: {expected_cases} ({mean_prob*100:.4f}%)")
        
    #     # Instead of using random numbers, force exactly the expected number of cases
    #     # Sort probabilities from highest to lowest
    #     sorted_indices = np.argsort(-probs)  # Negative sign for descending order
        
    #     # Take the top N indices as initial cases
    #     initial_case_indices = sorted_indices[:expected_cases]
    #     initial_cases = ss.uids(initial_case_indices)
        
    #     # Debug: Print initial cases
    #     print(f"Initial cases (deterministic): {len(initial_cases)} out of {n_people} ({len(initial_cases)/n_people*100:.4f}%)")
    #     print(f"Initial case indices: {initial_case_indices[:10]} (first 10 cases)")

    #     # Set prognoses for these cases
    #     self.set_prognoses(initial_cases)
        
    #     return initial_cases
    
    
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
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
            affected_sex=disease_params["affected_sex"]
        )
    
        # Define disease parameters
        self.define_pars(
            p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
        )
    
        # Define the lambda function to calculate acquisition probability
        def calculate_p_acquire(self, sim, uids):
            # Start with base probability
            p = np.full(len(uids), self.pars.p_acquire)
            
            # Apply sex-specific filtering
            if self.pars.affected_sex == "female":
                # Set probability to 0 for males
                p[sim.people.male[uids]] = 0
            elif self.pars.affected_sex == "male":
                # Set probability to 0 for females
                p[sim.people.female[uids]] = 0
                
            # Print base probabilities
            # print(f"Base probabilities (p): {p}")
            
            # Filter out invalid indices for HIV-specific relative susceptibility
            valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
            # Apply HIV-specific relative susceptibility
            p[valid_uids] *= self.pars.rel_sus_hiv
            # print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

            return p * self.rel_sus[uids]

        self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
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
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

        return new_cases

# Define Cervical Cancer condition
class CervicalCancer(ss.NCD):
    
    def __init__(self, disease_name='CervicalCancer', csv_path=None, pars=None, **kwargs):
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
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
            affected_sex=disease_params["affected_sex"]  # Cervical Cancer affects only females
        )
    
        # Define disease parameters
        self.define_pars(
            p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
        )
    
        # Define the lambda function to calculate acquisition probability
        def calculate_p_acquire(self, sim, uids):
            # Start with base probability
            p = np.full(len(uids), self.pars.p_acquire)
            
            # Apply sex-specific filtering
            if self.pars.affected_sex == "female":
                # Set probability to 0 for males
                p[sim.people.male[uids]] = 0
            elif self.pars.affected_sex == "male":
                # Set probability to 0 for females
                p[sim.people.female[uids]] = 0
                
            # Print base probabilities
            # print(f"Base probabilities (p): {p}")
            
            # Filter out invalid indices for HIV-specific relative susceptibility
            valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
            # Apply HIV-specific relative susceptibility
            p[valid_uids] *= self.pars.rel_sus_hiv
            # print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

            return p * self.rel_sus[uids]
        
        self.p_acquire = ss.bernoulli(p=calculate_p_acquire)
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
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

        return new_cases

# Define Prostate Cancer condition
class ProstateCancer(ss.NCD):
    
    def __init__(self, disease_name='ProstateCancer', csv_path=None, pars=None, **kwargs):
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
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  # Store HIV relative risk
            affected_sex=disease_params["affected_sex"]  # Prostate Cancer affects only males
        )
        
        # print(f"affected_sex is: {self.pars.affected_sex}")
    
        # Define disease parameters
        self.define_pars(
            p_acquire=disease_params["incidence"],  # Probability of acquisition per timestep
        )
    
        # Define the lambda function to calculate acquisition probability
        def calculate_p_acquire(self, sim, uids):
            # Start with base probability
            p = np.full(len(uids), self.pars.p_acquire)
            
            # Apply sex-specific filtering
            if self.pars.affected_sex == "female":
                # Set probability to 0 for males
                p[sim.people.male[uids]] = 0
            elif self.pars.affected_sex == "male":
                # Set probability to 0 for females
                p[sim.people.female[uids]] = 0
                
            # Print base probabilities
            # print(f"Base probabilities (p): {p}")
            
            # Filter out invalid indices for HIV-specific relative susceptibility
            valid_uids = [uid for uid in uids if uid in sim.people.hiv]
        
            # Apply HIV-specific relative susceptibility
            p[valid_uids] *= self.pars.rel_sus_hiv
            print(f"Probabilities after applying HIV-specific relative susceptibility (p): {p}")

            return p * self.rel_sus[uids]
        
        self.p_acquire = ss.bernoulli(p=calculate_p_acquire)

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
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

        return new_cases