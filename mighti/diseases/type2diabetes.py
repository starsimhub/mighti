"""
Module defining the Type 2 Diabetes remitting disease model.
"""


import starsim as ss
from mighti.diseases.base_disease import RemittingDisease
from starsim.interventions import treat_num


class Type2Diabetes(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type2Diabetes'
        super().__init__(csv_path, pars, **kwargs)

        self.define_pars(label='Type2Diabetes')  
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 0.018  
        return

   
class T2D_ReduceMortalityTx(treat_num):
    def __init__(self, *args, product=None, prob=1.0, rel_death_reduction=0.5, eligibility=None, **kwargs):
        super().__init__(*args, product=product, prob=prob, eligibility=eligibility, **kwargs)
        self.rel_death_reduction = rel_death_reduction
        self.disease = 'type2diabetes'  # fixed for T2D

    def initialize(self, sim):
        super().initialize(sim)

        # Set default eligibility to all affected if not given
        if self.eligibility is None:
            if not hasattr(sim.diseases, self.disease):
                raise ValueError(f"[{self.label}] Disease '{self.disease}' not found in sim.diseases.")
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids

    def step(self):
        self.add_to_queue()
        treat_inds = super().step()

        if len(treat_inds):
            successful = self.outcomes['successful']
            if len(successful):
                self.sim.diseases[self.disease].rel_death[ss.uids(successful)] *= self.rel_death_reduction
                print(f"[{self.label}] Successfully treated {len(successful)} T2D agents at step {self.ti}")
        return treat_inds
    