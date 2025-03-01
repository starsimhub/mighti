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

        for disease in self.diseases:
            self.age_bins[disease] = list(prevalence_data[disease]['male'].keys())
            self.age_bins[disease].sort()
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

            # Include susceptible tracking for infectious diseases (SIS model)
            if get_disease_class(disease) == "sis":
                self.results[f'{disease}_susceptible_male'] = np.zeros((npts, len(self.age_groups[disease])))
                self.results[f'{disease}_susceptible_female'] = np.zeros((npts, len(self.age_groups[disease])))

        self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years

        print(f"Initialized prevalence array with {npts} time points for {self.diseases}.")
        
        for disease in self.diseases:
            disease_obj = getattr(sim.diseases, disease.lower(), None)
            if disease_obj:
                disease_class = get_disease_class(disease)
                status_attr = 'infected' if disease_class == 'sis' else 'affected'

                # Print counts for debugging
                counts = {
                    "affected": np.sum(disease_obj.affected.raw) if hasattr(disease_obj, "affected") else "N/A",
                    "infected": np.sum(disease_obj.infected.raw) if hasattr(disease_obj, "infected") else "N/A",
                    "susceptible": np.sum(disease_obj.susceptible.raw) if hasattr(disease_obj, "susceptible") else "N/A"
                }
                # print(f"{disease}: Using `{status_attr}` | Counts: {counts}")

        return

    def step(self):
        sim = self.sim
        ages = sim.people.age[:]

        # Store single-age population distribution at each time step
        age_distribution = np.histogram(ages, bins=np.arange(0, 102, 1))
        self.results['population_age_distribution'][sim.ti, :] = age_distribution[0]

        for disease in self.diseases:
            disease_obj = getattr(sim.diseases, disease.lower())

            # Determine whether to use `infected` or `affected` based on disease classification
            disease_class = get_disease_class(disease)
            status_attr = 'infected' if disease_class == 'sis' else 'affected'

            # Retrieve prevalence data
            if hasattr(disease_obj, status_attr):
                status_array = getattr(disease_obj, status_attr)
            else:
                print(f"Warning: {status_attr} not found for {disease}, skipping.")
                continue

            for sex, label in zip([0, 1], ['male', 'female']):
                prevalence_by_age_group = np.zeros(len(self.age_groups[disease]))
                susceptible_by_age_group = np.zeros(len(self.age_groups[disease])) if disease_class == "sis" else None

                for i, (start, end) in enumerate(self.age_groups[disease]):
                    age_mask = (ages >= start) if end == float('inf') else (ages >= start) & (ages < end)

                    # Compute prevalence
                    status_for_age_group = status_array[:][age_mask]
                    if status_for_age_group.size > 0:
                        prevalence_by_age_group[i] = np.mean(status_for_age_group)

                    # Compute susceptible count if the disease is SIS
                    if disease_class == "sis" and hasattr(disease_obj, "susceptible"):
                        susceptible_for_age_group = disease_obj.susceptible[:][age_mask]
                        if susceptible_for_age_group.size > 0:
                            susceptible_by_age_group[i] = np.mean(susceptible_for_age_group)

                disease_key = f'{disease}_prevalence_{label}'
                self.results[disease_key][sim.ti, :] = prevalence_by_age_group

                if disease_class == "sis":
                    susceptible_key = f'{disease}_susceptible_{label}'
                    self.results[susceptible_key][sim.ti, :] = susceptible_by_age_group
      