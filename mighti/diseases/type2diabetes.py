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
        if not hasattr(self.pars, 'p_acquire_multiplier'):
            self.pars.p_acquire_multiplier = 0.092
        return


class ReduceMortalityTx(treat_num):
    def __init__(self, *args, product=None, prob=1.0, rel_death_reduction=0.5, eligibility=None, **kwargs):
        super().__init__(*args, product=product, prob=prob, eligibility=eligibility, **kwargs)
        self.rel_death_reduction = rel_death_reduction

    def step(self):
        self.add_to_queue()  # Fill queue
        treat_inds = super().step()  # Apply treatment using treat_num logic

        if len(treat_inds):
            successful = self.outcomes['successful']
            if len(successful):
                self.sim.diseases['type2diabetes'].rel_death[ss.uids(successful)] *= self.rel_death_reduction
                print(f"[{self.label}] Successfully treated {len(successful)} agents at step {self.ti}")
        return treat_inds
    