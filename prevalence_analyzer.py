import starsim as ss
import numpy as np
import sciris as sc

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

        print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
        return

    def apply(self, sim):
        print(f"Applying analyzer at time step {sim.ti}")
        ages = sim.people.age
        females = sim.people.female
    
        for disease in self.diseases:
            disease_obj = getattr(sim.diseases, disease.lower())
            if disease == 'HIV':
                status_attr = 'infected'
            else:
                status_attr = 'affected'
    
            for sex, label in zip([0, 1], ['male', 'female']):
                prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
    
                for i, (start, end) in enumerate(self.age_groups[disease]):
                    if end == float('inf'):
                        age_mask = (ages >= start) & (females == sex)
                    else:
                        age_mask = (ages >= start) & (ages < end) & (females == sex)
    
                    status_array = getattr(disease_obj, status_attr)
                    if np.sum(age_mask) > 0:
                        prevalence_by_age_group[i] = np.mean(status_array[age_mask])
    
                disease_key = f'{disease}_prevalence_{label}'
                # print(f"Storing data for {disease_key} at time {sim.ti}")  # Add this to confirm data is stored
                self.results[disease_key][sim.ti, :] = prevalence_by_age_group