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
        super().init_pre(sim)
        npts = len(sim.t)  # Number of time points in the simulation

        # Initialize result arrays for each disease: time x age groups
        for disease in self.diseases:
            self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))

            # Initialize result arrays for people living with and without HIV
            self.results[f'{disease}_prevalence_with_HIV_male'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_with_HIV_female'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_without_HIV_male'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_without_HIV_female'] = np.zeros((npts, len(self.age_groups[disease])))

        # Initialize array to store population age distribution for each year (single-age resolution)
        self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years (single-year resolution)

        print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
        return
     
    def step(self):
        sim = self.sim  # Access the sim object from the Analyzer base class
    
        # Extract ages of agents alive at this time step
        ages = sim.people.age[:]
        is_male = sim.people.male[:]
        hiv_status = sim.people.hiv.infected[:]
        
        print(f"ages: {ages}")
        print(f"uids: {is_male}")
        print(f"hiv_status: {hiv_status}")

        # Store single-age population distribution at each time step
        age_distribution = np.histogram(ages, bins=np.arange(0, 102, 1))  # Single-year resolution from 0 to 101
        self.results['population_age_distribution'][sim.ti, :] = age_distribution[0]
    
        # Existing logic for calculating and storing prevalence...
        for disease in self.diseases:
            disease_obj = getattr(sim.diseases, disease.lower())
            
            # Set 'infected' for HIV, HPV, and Flu; 'affected' for all other diseases
            status_attr = 'infected' if disease in ['HIV', 'HPV', 'Flu'] else 'affected'
            status_array = getattr(disease_obj, status_attr)

             # Identify the indices of entries that are not NaN in ages
            valid_age_indices = np.where(~np.isnan(ages))[0]
            print(f"np.where(~np.isnan(ages))[0]: {np.where(~np.isnan(ages))[0]}")
            
            # Use valid indices to filter ages, is_male, hiv_status, and uids
            hiv_status_alive = hiv_status[valid_age_indices]
            print(f"hiv_status_alive: {hiv_status_alive}")
            print(f"valid_age_indices: {valid_age_indices}")


            for sex, label in zip([0, 1], ['male', 'female']):
                prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
                prevalence_with_hiv_by_age_group = np.zeros(len(self.age_groups[disease]))
                prevalence_without_hiv_by_age_group = np.zeros(len(self.age_groups[disease]))

                for i, (start, end) in enumerate(self.age_groups[disease]):
                    age_mask = (ages >= start) if end == float('inf') else (ages >= start) & (ages < end)
                    sex_mask = (is_male == sex)
                    status_mask = age_mask & sex_mask
                    
                    status_for_group = status_array[:][status_mask]
                    hiv_status_for_group = hiv_status[status_mask]

                    if status_for_group.size > 0:
                        prevalence_by_age_group[i] = np.mean(status_for_group)
                        prevalence_with_hiv_by_age_group[i] = np.mean(status_for_group[hiv_status_for_group])
                        prevalence_without_hiv_by_age_group[i] = np.mean(status_for_group[~hiv_status_for_group])

                # Replace NaNs with zeros
                prevalence_by_age_group = np.nan_to_num(prevalence_by_age_group, nan=0.0)
                prevalence_with_hiv_by_age_group = np.nan_to_num(prevalence_with_hiv_by_age_group, nan=0.0)
                prevalence_without_hiv_by_age_group = np.nan_to_num(prevalence_without_hiv_by_age_group, nan=0.0)

                disease_key = f'{disease}_prevalence_{label}'
                self.results[disease_key][sim.ti, :] = prevalence_by_age_group

                disease_with_hiv_key = f'{disease}_prevalence_with_HIV_{label}'
                self.results[disease_with_hiv_key][sim.ti, :] = prevalence_with_hiv_by_age_group

                disease_without_hiv_key = f'{disease}_prevalence_without_HIV_{label}'
                self.results[disease_without_hiv_key][sim.ti, :] = prevalence_without_hiv_by_age_group

                # Debugging print statement
                print(f"Stored {disease_key}, {disease_with_hiv_key}, {disease_without_hiv_key} prevalence data for time index {sim.ti}")

    def finalize(self):
        super().finalize()
        # Debugging print statement to inspect the stored results
        print("Final stored results:")
        for key, value in self.results.items():
            print(f"{key}: {value}")
        return