import starsim as ss
import numpy as np
import sciris as sc

class PrevalenceAnalyzer(ss.Analyzer):
    """ Generalized analyzer to calculate disease prevalence over time by age group and sex """

    def __init__(self, prevalence_data, diseases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'prevalence_analyzer'
        self.prevalence_data = prevalence_data
        self.diseases = diseases  # List of diseases like ['HIV', 'type2diabetes', 'hypertension']

        # Initialize age bins for each disease
        self.age_bins = {}
        self.age_groups = {}

        for disease in self.diseases:
            self.age_bins[disease] = list(prevalence_data[disease]['male'].keys())
            self.age_bins[disease].sort()  # Ensure age bins are sorted
            self.age_groups[disease] = list(zip(self.age_bins[disease][:-1], self.age_bins[disease][1:])) + [(self.age_bins[disease][-1], float('inf'))]

        self.results = sc.odict()

    def init_pre(self, sim):
        """ Initialize storage for prevalence tracking """
        super().init_pre(sim)
        npts = sim.npts  # Number of time points in the simulation

        # Initialize result arrays for each disease
        for disease in self.diseases:
            self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))

        # Store population age distribution for each year (custom age bins)
        self.age_bins_custom = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  
        self.results['population_age_distribution'] = np.zeros((npts, len(self.age_bins_custom) - 1))

        print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
        return

    def apply(self, sim):
        """ Track prevalence and susceptible population at each time step """

        # Extract alive individuals
        alive_uids = np.where(sim.people.alive.raw)[0]
        ages = sim.people.age.raw[alive_uids]

        # Store population age distribution using custom age bins
        age_distribution, _ = np.histogram(ages, bins=self.age_bins_custom)
        self.results['population_age_distribution'][sim.ti, :] = age_distribution
        print(f"Population age distribution at time step {sim.ti}: {age_distribution}")

        # Print initial susceptible to T2D only at the start
        if sim.ti == 0 and 'type2diabetes' in sim.diseases:
            print(f"Initial susceptible to T2D: {np.sum(sim.diseases['type2diabetes'].susceptible.raw)}")

        # Track susceptible population for Type 2 Diabetes
        if 'type2diabetes' in sim.diseases:
            t2d_disease = sim.diseases['type2diabetes']
            susceptible_t2d = np.sum(t2d_disease.susceptible.raw)
            print(f"Time step {sim.ti}: Susceptible to T2D = {susceptible_t2d}")

        # Existing logic for calculating prevalence
        for disease in self.diseases:
            disease_obj = getattr(sim.diseases, disease.lower())
            status_attr = 'infected' if disease in ['HIV', 'HPV', 'Flu'] else 'affected'
            status_array = getattr(disease_obj, status_attr).raw  

            for sex, label in zip([0, 1], ['male', 'female']):
                prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))

                for i, (start, end) in enumerate(self.age_groups[disease]):
                    age_mask = (ages >= start) if end == float('inf') else (ages >= start) & (ages < end)
                    status_for_age_group = status_array[alive_uids][age_mask]
                    if status_for_age_group.size > 0:
                        prevalence_by_age_group[i] = np.mean(status_for_age_group)

                disease_key = f'{disease}_prevalence_{label}'
                self.results[disease_key][sim.ti, :] = prevalence_by_age_group


# import starsim as ss
# import numpy as np
# import sciris as sc



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

#         # Initialize array to store population age distribution for each year (single-age resolution)
#         self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years (single-year resolution)
#         # age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  # Your custom age bins
#         # self.results['population_age_distribution'] = np.zeros((npts, len(age_bins) - 1))  # Adjust to match bin count
#         # print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
#         return
        
#     def apply(self, sim):
#         # Print the age of a specific agent for debugging
#         # print(f'Age of agent 1: {sim.people.age[1]}, alive={sim.people.alive[1]}')
    
#         # Extract ages of agents alive at this time step
#         alive_uids = np.where(sim.people.alive.raw)[0]
#         ages = sim.people.age.raw[alive_uids]
    
#         # Store single-age population distribution at each time step
#         age_distribution = np.histogram(ages, bins=np.arange(0, 102, 1))  # Single-year resolution from 0 to 101
#         # age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  # Your custom age bins
#         # age_distribution = np.histogram(ages, bins=age_bins)
#         self.results['population_age_distribution'][sim.ti, :] = age_distribution[0]
#         # print(f"Population age distribution at time step {sim.ti}: {age_distribution[0]}")
#         # print(f"Population age distribution at time step {sim.ti} (by age bin): {age_distribution[0]}")
      
#         if 'type2diabetes' in sim.diseases:
#             t2d_disease = sim.diseases['type2diabetes']  # Get disease object
    
#             if hasattr(t2d_disease, 'susceptible'):  # Ensure it has the susceptible attribute
#                 susceptible_t2d = np.sum(t2d_disease.susceptible.raw)
#                 print(f"Time step {sim.ti}: Susceptible to T2D = {susceptible_t2d}")
#             else:
#                 print(f"Warning: Type2Diabetes does not have a 'susceptible' attribute.")
#         else:
#             print(f"Warning: Type2Diabetes not found in sim.diseases.")
            
            
#         if sim.ti == 0:  # Only print at the start
#             if 'type2diabetes' in sim.diseases:
#                 print(f"Initial susceptible to T2D: {np.sum(sim.diseases['type2diabetes'].susceptible.raw)}")
#             else:
#                 print("Warning: 'type2diabetes' not found in sim.diseases")
#             # Existing logic for calculating and storing prevalence...
#         for disease in self.diseases:
#             disease_obj = getattr(sim.diseases, disease.lower())
            
#             # Set 'infected' for HIV, HPV, and Flu; 'affected' for all other diseases
#             status_attr = 'infected' if disease in ['HIV', 'HPV', 'Flu'] else 'affected'
            
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
                

                
# import starsim as ss
# import numpy as np
# import sciris as sc


# class PrevalenceAnalyzer(ss.Analyzer):
#     """ Generalized analyzer to calculate disease prevalence over time by age group, sex, and HIV status. """

#     def __init__(self, prevalence_data, diseases, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = 'prevalence_analyzer'
#         self.prevalence_data = prevalence_data
#         self.diseases = diseases  # List of disease names like ['HIV', 'Type2Diabetes']

#         self.age_bins = {}
#         self.age_groups = {}

#         for disease in self.diseases:
#             self.age_bins[disease] = list(prevalence_data[disease]['male'].keys())
#             self.age_bins[disease].sort()
#             self.age_groups[disease] = list(zip(self.age_bins[disease][:-1], self.age_bins[disease][1:])) + [(self.age_bins[disease][-1], float('inf'))]

#         self.results = sc.odict()

#     def init_pre(self, sim):
#         super().init_pre(sim)
#         npts = sim.npts

#         for disease in self.diseases:
#             self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
#             self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))
            
#             # Track Type 2 Diabetes among HIV+ and HIV-
#             if disease == 'Type2Diabetes':
#                 self.results[f'{disease}_prevalence_hivpos_male'] = np.zeros((npts, len(self.age_groups[disease])))
#                 self.results[f'{disease}_prevalence_hivneg_male'] = np.zeros((npts, len(self.age_groups[disease])))
#                 self.results[f'{disease}_prevalence_hivpos_female'] = np.zeros((npts, len(self.age_groups[disease])))
#                 self.results[f'{disease}_prevalence_hivneg_female'] = np.zeros((npts, len(self.age_groups[disease])))

#         return

#     def apply(self, sim):
#         ages = sim.people.age
#         females = sim.people.female
#         hiv_status = sim.diseases.hiv.infected  # Boolean array: True if HIV+, False if HIV-
        
#         for disease in self.diseases:
#             disease_obj = getattr(sim.diseases, disease.lower())
#             status_attr = 'infected' if disease == 'HIV' else 'affected'
            
#             for sex, label in zip([0, 1], ['male', 'female']):
#                 prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
#                 if disease == 'Type2Diabetes':
#                     prevalence_hivpos_by_age_group = np.zeros(len(self.age_groups[disease]))
#                     prevalence_hivneg_by_age_group = np.zeros(len(self.age_groups[disease]))

#                 for i, (start, end) in enumerate(self.age_groups[disease]):
#                     if end == float('inf'):
#                         age_mask = (ages >= start) & (females == sex)
#                     else:
#                         age_mask = (ages >= start) & (ages < end) & (females == sex)

#                     status_array = getattr(disease_obj, status_attr)
#                     if np.sum(age_mask) > 0:
#                         prevalence_by_age_group[i] = np.mean(status_array[age_mask])

#                     if disease == 'Type2Diabetes':
#                         # Separate T2D prevalence for HIV+ and HIV- individuals
#                         prevalence_hivpos_by_age_group[i] = np.mean(status_array[age_mask & hiv_status]) if np.sum(age_mask & hiv_status) > 0 else 0
#                         prevalence_hivneg_by_age_group[i] = np.mean(status_array[age_mask & ~hiv_status]) if np.sum(age_mask & ~hiv_status) > 0 else 0

#                 self.results[f'{disease}_prevalence_{label}'][sim.ti, :] = prevalence_by_age_group

#                 if disease == 'Type2Diabetes':
#                     self.results[f'{disease}_prevalence_hivpos_{label}'][sim.ti, :] = prevalence_hivpos_by_age_group
#                     self.results[f'{disease}_prevalence_hivneg_{label}'][sim.ti, :] = prevalence_hivneg_by_age_group



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
#         print(f"Applying analyzer at time step {sim.ti}")
#         ages = sim.people.age
#         females = sim.people.female
    
#         for disease in self.diseases:
#             disease_obj = getattr(sim.diseases, disease.lower())
#             if disease == 'HIV':
#                 status_attr = 'infected'
#             else:
#                 status_attr = 'affected'
    
#             for sex, label in zip([0, 1], ['male', 'female']):
#                 prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
    
#                 for i, (start, end) in enumerate(self.age_groups[disease]):
#                     if end == float('inf'):
#                         age_mask = (ages >= start) & (females == sex)
#                     else:
#                         age_mask = (ages >= start) & (ages < end) & (females == sex)
    
#                     status_array = getattr(disease_obj, status_attr)
#                     if np.sum(age_mask) > 0:
#                         prevalence_by_age_group[i] = np.mean(status_array[age_mask])
    
#                 disease_key = f'{disease}_prevalence_{label}'
#                 # print(f"Storing data for {disease_key} at time {sim.ti}")  # Add this to confirm data is stored
#                 self.results[disease_key][sim.ti, :] = prevalence_by_age_group