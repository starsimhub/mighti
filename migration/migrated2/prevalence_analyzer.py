
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

        # Initialize age bins and groups for each disease
        self.age_bins = {}
        self.age_groups = {}
        for disease in self.diseases:
            self.age_bins[disease] = sorted(prevalence_data[disease]['male'].keys())  # Ensure age bins are sorted
            # Create age groups with "inf" for the last bin (80+)
            self.age_groups[disease] = list(zip(self.age_bins[disease][:-1], self.age_bins[disease][1:])) + [(self.age_bins[disease][-1], float('inf'))]

        self.results = sc.odict()

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = sim.t.npts  # Number of time points in the simulation

        # Initialize result arrays for each disease: time x age groups
        for disease in self.diseases:
            self.results[f'{disease}_prevalence_male'] = np.zeros((npts, len(self.age_groups[disease])))
            self.results[f'{disease}_prevalence_female'] = np.zeros((npts, len(self.age_groups[disease])))

        # Initialize array to store population age distribution for each year (single-age resolution)
        self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years (single-year resolution)

        print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
        return

    def step(self):
        sim = self.sim  # Access the sim object from the parent class

        # Extract ages of agents alive at this time step
        alive_uids = sim.people.alive.uids
        ages = sim.people.age[alive_uids]

        # Store single-age population distribution at each time step
        age_distribution = np.histogram(ages, bins=np.arange(0, 102, 1))  # Single-year resolution from 0 to 101
        self.results['population_age_distribution'][sim.t.ti, :] = age_distribution[0]

        # Calculate and store prevalence
        for disease in self.diseases:
            disease_obj = getattr(sim.diseases, disease.lower())

            # Set 'infected' for certain diseases; 'affected' for others
            status_attr = 'infected' if disease in ['HIV', 'HPV', 'Flu'] else 'affected'

            status_array = getattr(disease_obj, status_attr).values  # Use .values to access the data

            for sex, label in zip([0, 1], ['male', 'female']):
                sex_mask = (sim.people.female[alive_uids] == sex)  # Create mask for sex
                prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))

                for i, (start, end) in enumerate(self.age_groups[disease]):
                    age_mask = (ages >= start) if end == float('inf') else (ages >= start) & (ages < end)
                    combined_mask = sex_mask & age_mask

                    # Filter out relevant status values using the mask
                    status_for_age_group = status_array[alive_uids][combined_mask]
                    if status_for_age_group.size > 0:
                        prevalence_by_age_group[i] = np.mean(status_for_age_group)

                disease_key = f'{disease}_prevalence_{label}'
                self.results[disease_key][sim.t.ti, :] = prevalence_by_age_group
