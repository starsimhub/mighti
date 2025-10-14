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

        for uid in ppl.uid:
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
    pass

