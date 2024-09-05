import starsim as ss
import numpy as np
import sciris as sc

class PrevalenceAnalyzer(ss.Analyzer):
    """ Analyzer to calculate HIV and depression prevalence over time by age group """

    def __init__(self, age_data_hiv, age_data_depression, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'prevalence_analyzer'
        
        # Age data for HIV
        self.age_data_hiv = age_data_hiv
        self.age_bins_hiv = list(age_data_hiv.keys())  # Age bins for HIV
        self.age_groups_hiv = list(zip(self.age_bins_hiv[:-1], self.age_bins_hiv[1:]))  # Define HIV age groups

        # Age data for Depression
        self.age_data_depression = age_data_depression
        self.age_bins_depression = list(age_data_depression.keys())  # Age bins for depression
        self.age_groups_depression = list(zip(self.age_bins_depression[:-1], self.age_bins_depression[1:]))  # Define depression age groups

        self.results = sc.odict()  # Use an ordered dictionary to store the results

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = sim.npts  # Number of time points in the simulation

        # Initialize 2D arrays for both HIV and depression: time x age groups
        self.results['hiv_prevalence'] = np.zeros((npts, len(self.age_groups_hiv)))
        self.results['depression_prevalence'] = np.zeros((npts, len(self.age_groups_depression)))

        print(f"Initialized prevalence array with {npts} time points for both HIV and depression.")
        return

    def apply(self, sim):
        hiv = sim.diseases.hiv
        depression = sim.diseases.depression
        ages = sim.people.age
        
        # Calculate HIV prevalence by age group (based on HIV age bins)
        hiv_prevalence_by_age_group = np.zeros(len(self.age_groups_hiv))
        for i, (start, end) in enumerate(self.age_groups_hiv):
            age_mask = (ages >= start) & (ages < end)
            if np.sum(age_mask) > 0:
                hiv_prevalence_by_age_group[i] = np.mean(hiv.infected[age_mask])
        
        # Store HIV prevalence results
        self.results['hiv_prevalence'][sim.ti, :] = hiv_prevalence_by_age_group

        # Calculate Depression prevalence by age group (based on depression age bins)
        depression_prevalence_by_age_group = np.zeros(len(self.age_groups_depression))
        for i, (start, end) in enumerate(self.age_groups_depression):
            age_mask = (ages >= start) & (ages < end)
            if np.sum(age_mask) > 0:
                depression_prevalence_by_age_group[i] = np.mean(depression.affected[age_mask])
        
        # Store Depression prevalence results
        self.results['depression_prevalence'][sim.ti, :] = depression_prevalence_by_age_group
    