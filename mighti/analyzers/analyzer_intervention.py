"""
Analyzers for intervention 
"""

import numpy as np
import pandas as pd
import starsim as ss


__all__ = ["InterventionAnalyzer", "AdherenceAnalyzer"]


class InterventionAnalyzer(ss.Analyzer):
    """
    Tracks receipt of interventions for each agent over time.
    Handles both disease-linked interventions (e.g., ART via sim.diseases.hiv.on_art)
    and general intervention flags (e.g., ppl.intervention_history).
    """

    def __init__(self, interventions=None, **kwargs):
        super().__init__(**kwargs)
        self.name = 'intervention_analyzer'
        self.interventions = interventions or ['art', 'vmmc', 'prep', 'housing']
        self.records = []

    def init_results(self):
        super().init_results()
        self.records = []

    def step(self):
        sim = self.sim
        ppl = sim.people
        ti = sim.ti
        year = sim.t.yearvec[ti]

        for uid in ppl.auids:
            record = {
                'uid': uid,
                'year': year,
                'sex': 'Female' if ppl.female[uid] else 'Male',
                'age': ppl.age[uid],
            }

            for intv in self.interventions:
                received = False

                # 1. Check disease-linked intervention
                if hasattr(sim.diseases, 'hiv') and intv == 'art':
                    received = sim.diseases.hiv.on_art[uid]
                elif hasattr(ppl, 'intervention_history'):
                    # 2. Check generic intervention history (e.g., for housing or VMMC)
                    if intv in ppl.intervention_history:
                        received = ppl.intervention_history[intv][uid]

                record[f'received_{intv}'] = received

            self.records.append(record)

    def to_df(self):
        return pd.DataFrame(self.records)


class AdherenceAnalyzer(ss.Analyzer):
    """
    The share of people with HIV on a treatment, with and without a condition

    Args:
        condition_key (str): the state of the condition, as a key of `people.states`, e.g. "majordepressivedisorder.affected"
        intervention_key (str): the state of being on the treatment, e.g. "hiv.on_art"
    """
    def __init__(self, condition_key, intervention_key, label=None, **kwargs):
        super().__init__(label=label or f"on_{intervention_key}_by_{condition_key}", **kwargs)
        self.condition_key = condition_key
        self.intervention_key = intervention_key
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("on_with_condition", dtype=float, scale=False, label="On treatment, with the condition"),
            ss.Result("on_without_condition", dtype=float, scale=False, label="On treatment, without the condition"),
        )
        return

    def step(self):
        st = self.sim.people.states
        cond = np.asarray(st[self.condition_key].values, dtype=bool)
        on_tx = np.asarray(st[self.intervention_key].values, dtype=bool)
        hiv = np.asarray(st["hiv.infected"].values, dtype=bool) if "hiv.infected" in st else np.ones(len(cond), dtype=bool)
        for key, mask in [("on_with_condition", hiv & cond), ("on_without_condition", hiv & ~cond)]:
            self.results[key][self.ti] = on_tx[mask].mean() if mask.any() else np.nan
        return
