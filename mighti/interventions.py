"""
Implements interventions applied to the simulated population
"""


import starsim as ss


from starsim.interventions import treat_num


__all__ = ["ReduceMortalityTx"]


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
                # self.sim.diseases['type2diabetes'].rel_death[successful] *= self.rel_death_reduction
                self.sim.diseases['type2diabetes'].rel_death[ss.uids(successful)] *= self.rel_death_reduction
                print(f"[{self.label}] Successfully treated {len(successful)} agents at step {self.ti}")
        return treat_inds
    