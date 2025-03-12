import starsim as ss
import numpy as np
import sciris as sc
import pandas as pd

# Load disease classification from CSV
# csv_path = "mighti/data/eswatini_parameters.csv"
# df_params = pd.read_csv(csv_path, index_col="condition")

df_params = None  # Placeholder for external data

def initialize_prevalence_analyzer(data):
    """ Function to initialize prevalence analyzer with preloaded parameter data """
    global df_params
    df_params = data
    
def get_disease_class(condition):
    """Retrieve disease classification (e.g., 'ncd', 'disease', 'sis')"""
    try:
        return df_params.loc[condition, "disease_class"]
    except KeyError:
        print(f"Warning: Disease class not found for {condition}, defaulting to 'ncd'")
        return "ncd"

class PrevalenceAnalyzer(ss.Analyzer):
    """ Generalized analyzer to calculate disease prevalence over time by age group and sex """

    def __init__(self, prevalence_data, diseases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'prevalence_analyzer'
        self.prevalence_data = prevalence_data
        self.diseases = diseases  # List of disease names like ['HIV', 'depression', 'diabetes', ...]

        # Initialize age bins for each disease
        self.age_bins = {}
        self.age_groups = {}

        # Iterate over each disease and assign age bins
        for disease in self.diseases:
            self.age_bins[disease] = list(prevalence_data[disease]['male'].keys())
            self.age_bins[disease].sort()  # Ensure age bins are sorted
            # Create age groups with "inf" for the last bin (80+)
            self.age_groups[disease] = list(zip(self.age_bins[disease][:-1], self.age_bins[disease][1:])) + [(self.age_bins[disease][-1], float('inf'))]

        self.results = sc.objdict()

    def init_pre(self, sim):
        """ Initialize prevalence tracking and debug initial susceptible counts. """
        super().init_pre(sim)
        npts = len(sim.t)  # Number of time points in the simulation
    
        # Initialize result arrays for each disease: time x age groups
        for disease in self.diseases:
            self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_male_with_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_female_with_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_male_without_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_female_without_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
    
        self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years (single-year resolution)
    
        return

    def step(self):
        sim = self.sim  # Access the sim object from the Analyzer base class
    
        # Extract UIDs, ages, sex, and alive status of agents
        uids = sim.people.uid[:]
        ages = sim.people.age[:]
        is_male = sim.people.male[:]
        alive = sim.people.alive[:]  # Extract the alive status of agents
        hiv_status = sim.people.hiv.infected[:]  # Use 'infected' status for HIV
        print(f"ages: {ages}")
        print(f"uids: {uids}")
        print(f"alive: {alive}")
        print(f"is_male: {is_male}")
        print(f"hiv_status: {hiv_status}")
        print(f"ages: {len(ages)}")
        print(f"uids: {len(uids)}")
        print(f"alive: {len(alive)}")
        print(f"is_male: {len(is_male)}")
        print(f"hiv_status: {len(hiv_status)}")
        
        # Identify the indices of entries that are not NaN in ages
        valid_age_indices = np.where(~np.isnan(ages))[0]
        
        # Use valid indices to filter ages, is_male, hiv_status, and uids
        ages_alive = ages[valid_age_indices]
        is_male_alive = is_male[valid_age_indices]
        hiv_status_alive = hiv_status[valid_age_indices]
        uids_alive = uids[valid_age_indices]

        print(f"UIDs (alive): {uids_alive}")
        print(f"Ages (alive): {ages_alive}")
        print(f"Sex (alive): {is_male_alive}")
        print(f"HIV Status (alive): {hiv_status_alive}")

        # Store single-age population distribution at each time step
        age_distribution = np.histogram(ages_alive, bins=np.arange(0, 102, 1))  # Single-year resolution from 0 to 101
        self.results['population_age_distribution'][sim.ti, :] = age_distribution[0]
    
        # Existing logic for calculating and storing prevalence...
        for disease in self.diseases:
            disease_obj = getattr(sim.people, disease.lower())
            status_array = getattr(disease_obj, 'affected' if disease != 'HIV' else 'infected')[:]
            
            # Filter status_array to only include alive agents with valid ages
            status_array = status_array[valid_age_indices]
            
            for sex, label in zip([0, 1], ['male', 'female']):
                prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
                prevalence_by_age_group_with_HIV = np.zeros(len(self.age_groups[disease]))
                prevalence_by_age_group_without_HIV = np.zeros(len(self.age_groups[disease]))
    
                for i, (start, end) in enumerate(self.age_groups[disease]):
                    age_mask = (ages_alive >= start) if end == float('inf') else (ages_alive >= start) & (ages_alive < end)
                    sex_mask = (is_male_alive == sex)
                    status_mask = age_mask & sex_mask
                    
                    status_for_group = status_array[status_mask]
                    
                    if status_for_group.size > 0:
                        prevalence_by_age_group[i] = np.mean(status_for_group)
                    
                    status_mask_with_HIV = status_mask & hiv_status_alive
                    status_mask_without_HIV = status_mask & ~hiv_status_alive
                    status_for_group_with_HIV = status_array[status_mask_with_HIV]
                    status_for_group_without_HIV = status_array[status_mask_without_HIV]
                    
                    if status_for_group_with_HIV.size > 0:
                        prevalence_by_age_group_with_HIV[i] = np.mean(status_for_group_with_HIV)
                    
                    if status_for_group_without_HIV.size > 0:
                        prevalence_by_age_group_without_HIV[i] = np.mean(status_for_group_without_HIV)
    
                disease_key = f'{disease}_prevalence_{label}'
                self.results[disease_key][sim.ti, :] = prevalence_by_age_group
                self.results[f'{disease}_prevalence_{label}_with_HIV'][sim.ti, :] = prevalence_by_age_group_with_HIV
                self.results[f'{disease}_prevalence_{label}_without_HIV'][sim.ti, :] = prevalence_by_age_group_without_HIV
                
# import starsim as ss
# import numpy as np
# import sciris as sc
# import pandas as pd

# # Load disease classification from CSV
# # csv_path = "mighti/data/eswatini_parameters.csv"
# # df_params = pd.read_csv(csv_path, index_col="condition")

# df_params = None  # Placeholder for external data

# def initialize_prevalence_analyzer(data):
#     """ Function to initialize prevalence analyzer with preloaded parameter data """
#     global df_params
#     df_params = data
    
# def get_disease_class(condition):
#     """Retrieve disease classification (e.g., 'ncd', 'disease', 'sis')"""
#     try:
#         return df_params.loc[condition, "disease_class"]
#     except KeyError:
#         print(f"Warning: Disease class not found for {condition}, defaulting to 'ncd'")
#         return "ncd"

# class PrevalenceAnalyzer(ss.Analyzer):
#     """ Generalized analyzer to calculate disease prevalence over time by age group and sex """

#     def __init__(self, prevalence_data, diseases, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = 'prevalence_analyzer'
#         self.prevalence_data = prevalence_data
#         self.diseases = diseases  # List of disease names like ['HIV', 'depression', 'diabetes', ...]

#         # Initialize age bins for each disease
#         self.age_bins = {}
#         self.age_groups = {}

#         # Iterate over each disease and assign age bins
#         for disease in self.diseases:
#             self.age_bins[disease] = list(prevalence_data[disease]['male'].keys())
#             self.age_bins[disease].sort()  # Ensure age bins are sorted
#             # Create age groups with "inf" for the last bin (80+)
#             self.age_groups[disease] = list(zip(self.age_bins[disease][:-1], self.age_bins[disease][1:])) + [(self.age_bins[disease][-1], float('inf'))]

#         self.results = sc.objdict()

#     def init_pre(self, sim):
#         """ Initialize prevalence tracking and debug initial susceptible counts. """
#         super().init_pre(sim)
#         npts = len(sim.t)  # Number of time points in the simulation
    
#         # Initialize result arrays for each disease: time x age groups
#         for disease in self.diseases:
#             self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
#             self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))
#             self.results[f'{disease}_prevalence_male_with_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
#             self.results[f'{disease}_prevalence_female_with_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
#             self.results[f'{disease}_prevalence_male_without_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
#             self.results[f'{disease}_prevalence_female_without_HIV'] = np.zeros((npts, len(self.age_groups[disease])))
    
#         self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years (single-year resolution)
    
#         return

#     def step(self):
#         sim = self.sim  # Access the sim object from the Analyzer base class
    
#         # Extract ages of agents alive at this time step
#         ages = sim.people.age
#         is_male = sim.people.male
#         hiv_status = sim.people.hiv.infected[:]  # Use 'infected' status for HIV
        
#         # Store single-age population distribution at each time step
#         age_distribution = np.histogram(ages, bins=np.arange(0, 102, 1))  # Single-year resolution from 0 to 101
#         self.results['population_age_distribution'][sim.ti, :] = age_distribution[0]
    
#         # Existing logic for calculating and storing prevalence...
#         for disease in self.diseases:
#             disease_obj = getattr(sim.people, disease.lower())
#             status_array = getattr(disease_obj, 'affected' if disease != 'HIV' else 'infected')[:]
  

#             for sex, label in zip([0, 1], ['male', 'female']):
#                 prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
#                 prevalence_by_age_group_with_HIV = np.zeros(len(self.age_groups[disease]))
#                 prevalence_by_age_group_without_HIV = np.zeros(len(self.age_groups[disease]))

    
#                 for i, (start, end) in enumerate(self.age_groups[disease]):
#                     age_mask = (ages >= start) if end == float('inf') else (ages >= start) & (ages < end)
#                     sex_mask = (is_male == sex)
#                     status_mask = age_mask & sex_mask
#                     status_for_group = status_array[status_mask]
                    
#                     if status_for_group.size > 0:
#                         prevalence_by_age_group[i] = np.mean(status_for_group)
#                         status_mask_with_HIV = status_mask & hiv_status
#                         status_mask_without_HIV = status_mask & ~hiv_status
#                         status_for_group_with_HIV = status_array[status_mask_with_HIV]
#                         status_for_group_without_HIV = status_array[status_mask_without_HIV] 
                    
#                     if status_for_group_with_HIV.size > 0:
#                         prevalence_by_age_group_with_HIV[i] = np.mean(status_for_group_with_HIV)
                    
#                     if status_for_group_without_HIV.size > 0:
#                         prevalence_by_age_group_without_HIV[i] = np.mean(status_for_group_without_HIV)
    
#                 disease_key = f'{disease}_prevalence_{label}'
#                 self.results[disease_key][sim.ti, :] = prevalence_by_age_group
#                 self.results[f'{disease}_prevalence_{label}_with_HIV'][sim.ti, :] = prevalence_by_age_group_with_HIV
#                 self.results[f'{disease}_prevalence_{label}_without_HIV'][sim.ti, :] = prevalence_by_age_group_without_HIV