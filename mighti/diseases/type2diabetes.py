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

        self.define_pars(label='type2diabetes')  
        return

   
class T2D_ReduceMortalityTx(treat_num):
    def __init__(self, *args, product=None, prob=1.0, rel_death_reduction=0.5, eligibility=None, **kwargs):
        super().__init__(*args, product=product, prob=prob, eligibility=eligibility, **kwargs)
        self.rel_death_reduction = rel_death_reduction
        self.disease = 'type2diabetes'  

    def initialize(self, sim):
        super().initialize(sim)

        if self.eligibility is None:
            if not hasattr(sim.diseases, self.disease):
                raise ValueError(f"[{self.label}] Disease '{self.disease}' not found in sim.diseases.")
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids
    
    def init_pre(self, sim):
        super().init_pre(sim)
        self._budget_module = sim.get_module("budget_constraint", optional=True)

    def apply(self, sim):
        super().apply(sim)  # normal intervention behavior

        if self._budget_module:
            n_treated = getattr(self, "n_treated", 0)
            cost_ppy = getattr(self, "cost_per_person_year", 0.0) or 0.0
            cost = n_treated * float(cost_ppy) / sim.n_years
            hrh_minutes = dict(doctor=5 * n_treated, nurse=30 * n_treated)
            self._budget_module.register_usage(cost=cost, hrh_minutes=hrh_minutes, source=self.name)

    def step(self):
        self.add_to_queue()
        treat_inds = super().step()

        if len(treat_inds):
            successful = self.outcomes['successful']
            if len(successful):
                self.sim.diseases[self.disease].rel_death[ss.uids(successful)] *= self.rel_death_reduction
        return treat_inds
    