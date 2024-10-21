import starsim as ss
import numpy as np
import sciris as sc

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

#         self.results = sc.odict()

#     def init_pre(self, sim):
#         super().init_pre(sim)
#         npts = sim.npts  # Number of time points in the simulation

#         # Initialize result arrays for each disease: time x age groups
#         for disease in self.diseases:
#             self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
#             self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))

#         print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
#         return


    
#     def apply(self, sim):
#         # Print the age of a specific agent for debugging
#         print(f'Age of agent 100: {sim.people.age[100]}, alive={sim.people.alive[100]}')
    
#         # Extract ages of agents alive at this time step
#         alive_uids = np.where(sim.people.alive.raw)[0]
#         ages = sim.people.age.raw[alive_uids]
    
#         # Store age distribution for the current time step
#         age_distribution = np.histogram(ages, bins=np.arange(0, 101, 5))  # Example with 5-year bins
#         print(f"Age distribution at time step {sim.ti}: {age_distribution[0]}")
    
#         # Existing logic for calculating and storing prevalence...
#         for disease in self.diseases:
#             disease_obj = getattr(sim.diseases, disease.lower())
#             status_attr = 'infected' if disease == 'HIV' else 'affected'
#             status_array = getattr(disease_obj, status_attr).raw  # Use .raw to extract the underlying data
    
#             for sex, label in zip([0, 1], ['male', 'female']):
#                 prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
    
#                 for i, (start, end) in enumerate(self.age_groups[disease]):
#                     age_mask = (ages >= start) if end == float('inf') else (ages >= start) & (ages < end)
                    
#                     # Filter out relevant status values using the mask
#                     status_for_age_group = status_array[alive_uids][age_mask]
#                     if status_for_age_group.size > 0:
#                         prevalence_by_age_group[i] = np.mean(status_for_age_group)
    
#                 disease_key = f'{disease}_prevalence_{label}'
#                 self.results[disease_key][sim.ti, :] = prevalence_by_age_group



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

        self.results = sc.odict()

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = sim.npts  # Number of time points in the simulation

        # Initialize result arrays for each disease: time x age groups
        for disease in self.diseases:
            self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))

        # Initialize array to store population age distribution for each year (single-age resolution)
        self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years (single-year resolution)

        print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
        return

    
    def apply(self, sim):
        # Print the age of a specific agent for debugging
        # print(f'Age of agent 100: {sim.people.age[100]}, alive={sim.people.alive[100]}')
    
        # Extract ages of agents alive at this time step
        alive_uids = np.where(sim.people.alive.raw)[0]
        ages = sim.people.age.raw[alive_uids]
    
        # Store single-age population distribution at each time step
        age_distribution = np.histogram(ages, bins=np.arange(0, 102, 1))  # Single-year resolution from 0 to 101
        self.results['population_age_distribution'][sim.ti, :] = age_distribution[0]
        # print(f"Population age distribution at time step {sim.ti}: {age_distribution[0]}")
    
        # Existing logic for calculating and storing prevalence...
        for disease in self.diseases:
            disease_obj = getattr(sim.diseases, disease.lower())
            status_attr = 'infected' if disease == 'HIV' else 'affected'
            status_array = getattr(disease_obj, status_attr).raw  # Use .raw to extract the underlying data
    
            for sex, label in zip([0, 1], ['male', 'female']):
                prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
    
                for i, (start, end) in enumerate(self.age_groups[disease]):
                    age_mask = (ages >= start) if end == float('inf') else (ages >= start) & (ages < end)
                    
                    # Filter out relevant status values using the mask
                    status_for_age_group = status_array[alive_uids][age_mask]
                    if status_for_age_group.size > 0:
                        prevalence_by_age_group[i] = np.mean(status_for_age_group)
    
                disease_key = f'{disease}_prevalence_{label}'
                self.results[disease_key][sim.ti, :] = prevalence_by_age_group