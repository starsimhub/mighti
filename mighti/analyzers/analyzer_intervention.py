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
    def __init__(self, condition_key, intervention_key, label=None):
        """
        condition_key: str — e.g., "majordepressivedisorder.affected"
        intervention_key: str — e.g., "hiv.on_art"
        """
        super().__init__()
        self.condition_key = condition_key
        self.intervention_key = intervention_key
        self.label = label or f"on_{intervention_key}_by_{condition_key}"

    def initialize(self, sim):
        self.results = {
            "time": [],
            "on_with_condition": [],
            "on_without_condition": [],
        }

    def apply(self, sim):
        st = sim.people.states
        cond = st.get(self.condition_key, np.zeros(sim.n_agents, dtype=bool))
        on_tx = st.get(self.intervention_key, np.zeros(sim.n_agents, dtype=bool))

        is_alive = st.get("alive", np.ones(sim.n_agents, dtype=bool))
        eligible = is_alive & st.get("hiv.infected", np.ones(sim.n_agents, dtype=bool))  # modify if needed

        cond_alive = eligible & cond
        notcond_alive = eligible & ~cond

        def safe_mean(arr1, arr2):
            return np.nan if arr2.sum() == 0 else arr1[arr2].mean()

        self.results["time"].append(sim.t)
        self.results["on_with_condition"].append(safe_mean(on_tx, cond_alive))
        self.results["on_without_condition"].append(safe_mean(on_tx, notcond_alive))

    def finalize(self, sim):
        sim.results[self.label] = self.results
