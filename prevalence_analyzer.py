import starsim as ss
import numpy as np
import sciris as sc

class PrevalenceAnalyzer(ss.Analyzer):
    """ Analyzer to calculate HIV prevalence over time by age group """

    def __init__(self, age_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'hiv_prevalence'
        self.age_data = age_data
        self.age_groups = list(zip(age_data.keys(), list(age_data.keys())[1:]))
        self.results = sc.odict()  # Use an ordered dictionary to store the results

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = sim.npts  # Number of time points in the simulation
        n_age_groups = len(self.age_groups)
        self.results['prevalence'] = np.zeros((npts, n_age_groups))  # 2D array: time x age groups
        print(f"Initialized prevalence array with {npts} time points and {n_age_groups} age groups.")
        return

    def apply(self, sim):
        """ Calculate the HIV prevalence at the current timestep for each age group """
        hiv = sim.diseases.hiv
        ages = sim.people.age
        prevalence_by_age_group = np.zeros(len(self.age_groups))
        
        for i, (start, end) in enumerate(self.age_groups):
            age_mask = (ages >= start) & (ages < end)
            if np.sum(age_mask) > 0:
                prevalence_by_age_group[i] = np.mean(hiv.infected[age_mask])
            else:
                prevalence_by_age_group[i] = 0  # Handle cases where no one is in this age group

        self.results['prevalence'][sim.ti, :] = prevalence_by_age_group  # Store the prevalence for all age groups
        print(f"Time step {sim.ti}: Prevalence by age group = {prevalence_by_age_group}")
        return

    