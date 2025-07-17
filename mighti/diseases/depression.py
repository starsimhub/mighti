"""
Module defining depressin as a remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease
from starsim.interventions import treat_num


class Depression(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Depression'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'Depression')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 0.018  
        return


class DepressionCare(treat_num):
    def __init__(self, product=None, prob=1.0, max_capacity=None, eligibility=None, label='DepressionCare', **kwargs):
        super().__init__(product=product, prob=prob, max_capacity=max_capacity, eligibility=eligibility, label=label, **kwargs)
        self.disease = 'depression'  # used for default eligibility

    def initialize(self, sim):
        super().initialize(sim)
        # If no custom eligibility is given, treat all affected
        if self.eligibility is None:
            if not hasattr(sim.diseases, self.disease):
                raise ValueError(f"[{self.label}] Disease '{self.disease}' not found in sim.diseases.")
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids

    def step(self):
        self.add_to_queue()
        treat_inds = super().step()
        if len(treat_inds):
            print(f"[{self.label}] Treated {len(treat_inds)} agents for depression at step {self.ti}")
        return treat_inds
    
