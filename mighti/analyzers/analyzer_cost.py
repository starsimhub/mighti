import starsim as ss
import numpy as np
import pandas as pd
import sciris as sc


class MicrocostingAnalyzer(ss.Analyzer):
    def __init__(self, unit_costs=None, disability_weights=None, discount_rate=0.03, **kwargs):
        super().__init__(**kwargs)
        self.name = 'microcostinganalyzer'

        # Input parameters
        self.unit_costs = unit_costs or {}  # e.g., {'hospitalization': 200, 'art': 50}
        self.disability_weights = disability_weights or {}  # e.g., {'HIV': 0.2}
        self.discount_rate = discount_rate

        # Internal storage
        self.detailed_outputs = None
        return

    def init_results(self):
        self.records = []

    def step(self):
        # Not used for now (post-simulation analysis)
        pass

    def finalize(self):
        super().finalize()

        ppl = self.sim.people
        n = len(ppl)
        uids = ppl.uid

        # Set up arrays
        total_cost = np.zeros(n)
        total_yld = np.zeros(n)
        cost_details = {}
        yld_details = {}

        # -------------------------
        # Event-based costing (e.g., hospitalizations)
        # -------------------------
        for event, unit_cost in self.unit_costs.items():
            if event == 'art':
                continue  # ART handled separately below

            if hasattr(ppl, f'{event}_count'):
                event_counts = getattr(ppl, f'{event}_count')
                if isinstance(event_counts, dict):
                    counts = np.array([event_counts.get(uid, 0) for uid in uids])
                else:
                    counts = event_counts
                cost = counts * unit_cost
                cost_details[f'{event}_cost'] = cost
                total_cost += cost

        # -------------------------
        # Disability weights → YLD
        # -------------------------
        for cond, weight in self.disability_weights.items():
            if hasattr(ppl, cond) and hasattr(ppl[cond], 'duration'):
                durations = ppl[cond].duration
                yld = durations * weight
                yld_details[f'{cond}_yld'] = yld
                total_yld += yld

        # -------------------------
        # Intervention-based costs (e.g., ART)
        # -------------------------
        try:
            intervention_analyzer = self.sim.analyzers.intervention_analyzer
        except AttributeError:
            raise ValueError("MicrocostingAnalyzer requires InterventionAnalyzer with name='intervention_analyzer'.")
        
        if 'art' in self.unit_costs:
            art_df = intervention_analyzer.to_df()
            # Count # of years each person received ART
            art_counts = (
                art_df[art_df['received_art']]
                .groupby('uid')
                .size()
                .reindex(uids, fill_value=0)
                .values
            )
            art_cost = art_counts * self.unit_costs['art']
            cost_details['art_cost'] = art_cost
            total_cost += art_cost

        # -------------------------
        # Combine into DataFrame
        # -------------------------
        df = pd.DataFrame({
            'uid': uids,
            'total_cost': total_cost,
            'total_yld': total_yld,
        })

        for key, arr in cost_details.items():
            df[key] = arr

        for key, arr in yld_details.items():
            df[key] = arr

        self.detailed_outputs = df
        return

    def to_df(self):
        return self.detailed_outputs