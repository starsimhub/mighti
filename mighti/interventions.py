"""
Implements interventions applied to the simulated population
"""


import starsim as ss
import numpy as np


from starsim.interventions import treat_num

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
    
# class BaseHealthIntervention(ss.Intervention):
#     def __init__(self, name=None, start=None, stop=None, uptake_prob=1.0, eligibility=None, **kwargs):
#         super().__init__(name=name, **kwargs)

#         self.start = start
#         self.stop = stop
#         self.uptake_prob = ss.bernoulli(uptake_prob)
#         self.eligibility = eligibility if eligibility is not None else ss.uids()

#         self.define_states(
#             ss.State('intervened'),
#             ss.FloatArr('ti_intervened'),
#         )

#         self.define_results(
#             ss.Result('n_intervened', dtype=int, label="Received intervention"),
#         )

#     def get_recipients(self):
#         if callable(self.eligibility):
#             eligible = self.eligibility(self.sim)
#         else:
#             eligible = self.eligibility
#         return self.uptake_prob.filter(eligible)

#     def apply_effects(self, uids):
#         """To be overridden in subclass"""
#         pass

#     def step(self):
#         if self.start and self.t.now('year') < self.start:
#             return
#         if self.stop and self.t.now('year') >= self.stop:
#             return

#         uids = self.get_recipients()
#         if len(uids):
#             self.intervened[uids] = True
#             self.ti_intervened[uids] = self.ti
#             self.apply_effects(uids)

#     def update_results(self):
#         self.results['n_intervened'][self.ti] = np.count_nonzero(self.ti_intervened == self.ti)
        

# class DepressionDrugIntervention(BaseHealthIntervention):
#     def __init__(self, name='depression_tx', start=None, stop=None, uptake_prob=1.0, **kwargs):
#         super().__init__(name=name, start=start, stop=stop, uptake_prob=uptake_prob, **kwargs)

#     def apply_effects(self, uids):
#         """Flag agents as receiving depression medication"""
#         self.sim.conditions['Depression'].on_treatment[uids] = True
        