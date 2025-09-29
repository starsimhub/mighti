import starsim as ss
import numpy as np
import pandas as pd
import sciris as sc


class MicrocostingAnalyzer(ss.Analyzer):
    def __init__(self, unit_costs=None, disability_weights=None,
                  discount_rate=None,  # ← legacy arg support
                  discount_rate_costs=0.03, discount_rate_outcomes=0.03,
                  **kwargs):
        
        # Backward compatibility: use `discount_rate` if provided
        if discount_rate is not None:
            discount_rate_costs = discount_rate
            discount_rate_outcomes = discount_rate
    
        # Store parameters
        self.unit_costs = unit_costs or {}
        self.disability_weights = disability_weights or {}
        self.discount_rate_costs = discount_rate_costs
        self.discount_rate_outcomes = discount_rate_outcomes
    
        # Remove from kwargs to avoid passing to StarSim base class
        kwargs.pop('discount_rate', None)
        kwargs.pop('discount_rate_costs', None)
        kwargs.pop('discount_rate_outcomes', None)
    
        super().__init__(**kwargs)
        self.name = 'microcostinganalyzer'
        self.detailed_outputs = None

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
        years = self.sim.t.yearvec
        n_years = len(years)
    
        print(f"\n📊 Finalizing MicrocostingAnalyzer for {n:,} agents across {n_years} years")
    
        # Arrays
        total_cost = np.zeros(n)
        total_yld = np.zeros(n)
        total_yll = np.zeros(n)
        cost_details = {}
        yld_details = {}
        yll_details = {}
    
        # -------------------------
        # Event-based costing
        # -------------------------
        print("\n💵 Event-based costs:")
        for event, unit_cost in self.unit_costs.items():
            if event == 'art':
                continue
            if hasattr(ppl, f'{event}_count'):
                event_counts = getattr(ppl, f'{event}_count')
                if isinstance(event_counts, dict):
                    counts = np.array([event_counts.get(uid, 0) for uid in uids])
                else:
                    counts = event_counts
                cost = counts * unit_cost / ((1 + self.discount_rate_costs) ** (n_years - 1))
                print(f"  • {event}: {cost.sum():,.2f}")
                cost_details[f'{event}_cost'] = cost
                total_cost += cost
    
        # -------------------------
        # YLDs from condition durations
        # -------------------------
        print("\n🧮 YLDs by condition:")
        for cond, weight in self.disability_weights.items():
            if hasattr(ppl, cond) and hasattr(ppl[cond], 'duration'):
                durations = ppl[cond].duration
                yld = durations * weight / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                yld_details[f'{cond}_yld'] = yld
                total_yld += yld
                print(f"  • {cond}: {yld.sum():.2f}")
            else:
                print(f"  ⚠️ Missing: {cond} or duration attribute not found")
    
        # -------------------------
        # YLLs from external analyzer
        # -------------------------
        print("\n🧮 YLLs from ConditionAtDeathAnalyzer:")
        condition_death = self.sim.analyzers.condition_at_death_analyzer
        if condition_death and hasattr(condition_death, 'to_df'):
            print(f" Number of deaths recorded: {len(condition_death.records)}")
            df_yll = condition_death.to_df()
            if len(df_yll):
                print("📋 ConditionAtDeathAnalyzer DataFrame (preview):")
                print(df_yll[['uid', 'year', 'age', 'sex', 'yll']].head())
            # Only calculate YLLs for those with death records
            death_uids = df_yll['uid'].values
            yll_array = df_yll['yll'].values
            yll_array_discounted = yll_array / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
            print(f" yll_array_discounted: {yll_array_discounted}")
            print(f"uids dtype: {uids.dtype}, unique: {uids[:5]}")
            # Create a Series mapping uid → yll_discounted
            discounted_series = pd.Series(yll_array_discounted, index=death_uids)
            print(f"discounted_series.index dtype: {discounted_series.index.dtype}, unique: {discounted_series.index[:5]}")
            print(f"Overlap in UID sets: {len(set(uids) & set(discounted_series.index))}")
            print(f" discounted_series: {discounted_series}")
            # Map the full uids array
            mapped = discounted_series.reindex(np.arange(ppl.n_uids), fill_value=0.0).values
            print(f" mapped {mapped}")
            # Now add to total_yll
            total_yll += mapped
            yll_details['yll'] = mapped
            print(f"  • Total YLLs: {mapped.sum():.2f}")
        else:
            print("  ⚠️ ConditionAtDeathAnalyzer not found")
    
        # -------------------------
        # ART costs via InterventionAnalyzer
        # -------------------------
        print("\n💊 ART costing:")
        try:
            intervention_analyzer = self.sim.analyzers.intervention_analyzer
        except AttributeError:
            raise ValueError("MicrocostingAnalyzer requires InterventionAnalyzer named 'intervention_analyzer'.")
    
        if 'art' in self.unit_costs:
            art_df = intervention_analyzer.to_df()
            art_counts = (
                art_df[art_df['received_art']]
                .groupby('uid').size()
                .reindex(uids, fill_value=0)
                .values
            )
            art_cost = art_counts * self.unit_costs['art'] / ((1 + self.discount_rate_costs) ** (n_years - 1))
            cost_details['art_cost'] = art_cost
            total_cost += art_cost
            print(f"  • ART total cost: {art_cost.sum():,.2f} for {art_counts.sum():,} doses")
    
        # -------------------------
        # Output dataframe
        # -------------------------
        df = pd.DataFrame({
            'uid': uids,
            'total_cost': total_cost,
            'total_yld': total_yld,
            'total_yll': total_yll,
            'total_daly': total_yld + total_yll,
        })
    
        for key, val in cost_details.items():
            df[key] = val
        for key, val in yld_details.items():
            df[key] = val
        for key, val in yll_details.items():
            df[key] = val
    
        self.detailed_outputs = df
    
        print(f"\n✅ Finalized MicrocostingAnalyzer")
        print(f"   → Total cost: ${total_cost.sum():,.2f}")
        print(f"   → Total YLD: {total_yld.sum():,.2f}")
        print(f"   → Total YLL: {total_yll.sum():,.2f}")
        print(f"   → Total DALY: {df['total_daly'].sum():,.2f}")
    
        return

    def compute_icer(self, other_analyzer):
        df_self = self.to_df().set_index('uid')
        df_other = other_analyzer.to_df().set_index('uid')
        df_common = df_self.join(df_other, lsuffix='_intv', rsuffix='_base', how='inner')

        delta_cost = df_common['total_cost_intv'].sum() - df_common['total_cost_base'].sum()
        delta_daly = df_common['total_daly_base'].sum() - df_common['total_daly_intv'].sum()

        icer = delta_cost / delta_daly if delta_daly != 0 else np.inf
        return {
            'delta_cost': delta_cost,
            'delta_daly': delta_daly,
            'icer': icer,
        }

    def to_df(self):
        return self.detailed_outputs
    
    
    
# import starsim as ss
# import numpy as np
# import pandas as pd
# import sciris as sc


# class MicrocostingAnalyzer(ss.Analyzer):
#     def __init__(self, unit_costs=None, disability_weights=None, discount_rate=0.03, **kwargs):
#         super().__init__(**kwargs)
#         self.name = 'microcostinganalyzer'

#         # Input parameters
#         self.unit_costs = unit_costs or {}  # e.g., {'hospitalization': 200, 'art': 50}
#         self.disability_weights = disability_weights or {}  # e.g., {'HIV': 0.2}
#         self.discount_rate = discount_rate

#         # Internal storage
#         self.detailed_outputs = None
#         return

#     def init_results(self):
#         self.records = []

#     def step(self):
#         # Not used for now (post-simulation analysis)
#         pass

#     def finalize(self):
#         super().finalize()

#         ppl = self.sim.people
#         n = len(ppl)
#         uids = ppl.uid

#         # Set up arrays
#         total_cost = np.zeros(n)
#         total_yld = np.zeros(n)
#         cost_details = {}
#         yld_details = {}

#         # -------------------------
#         # Event-based costing (e.g., hospitalizations)
#         # -------------------------
#         for event, unit_cost in self.unit_costs.items():
#             if event == 'art':
#                 continue  # ART handled separately below

#             if hasattr(ppl, f'{event}_count'):
#                 event_counts = getattr(ppl, f'{event}_count')
#                 if isinstance(event_counts, dict):
#                     counts = np.array([event_counts.get(uid, 0) for uid in uids])
#                 else:
#                     counts = event_counts
#                 cost = counts * unit_cost
#                 cost_details[f'{event}_cost'] = cost
#                 total_cost += cost

#         # -------------------------
#         # Disability weights → YLD
#         # -------------------------
#         for cond, weight in self.disability_weights.items():
#             if hasattr(ppl, cond) and hasattr(ppl[cond], 'duration'):
#                 durations = ppl[cond].duration
#                 yld = durations * weight
#                 yld_details[f'{cond}_yld'] = yld
#                 total_yld += yld

#         # -------------------------
#         # Intervention-based costs (e.g., ART)
#         # -------------------------
#         try:
#             intervention_analyzer = self.sim.analyzers.intervention_analyzer
#         except AttributeError:
#             raise ValueError("MicrocostingAnalyzer requires InterventionAnalyzer with name='intervention_analyzer'.")
        
#         if 'art' in self.unit_costs:
#             art_df = intervention_analyzer.to_df()
#             # Count # of years each person received ART
#             art_counts = (
#                 art_df[art_df['received_art']]
#                 .groupby('uid')
#                 .size()
#                 .reindex(uids, fill_value=0)
#                 .values
#             )
#             art_cost = art_counts * self.unit_costs['art']
#             cost_details['art_cost'] = art_cost
#             total_cost += art_cost

#         # -------------------------
#         # Combine into DataFrame
#         # -------------------------
#         df = pd.DataFrame({
#             'uid': uids,
#             'total_cost': total_cost,
#             'total_yld': total_yld,
#         })

#         for key, arr in cost_details.items():
#             df[key] = arr

#         for key, arr in yld_details.items():
#             df[key] = arr

#         self.detailed_outputs = df
#         return
    
    
#     def compute_icer(self, other_analyzer):
#         """
#         Compute ICER relative to another simulation.
    
#         Args:
#             other_analyzer (MicrocostingAnalyzer): The baseline or comparison analyzer.
    
#         Returns:
#             dict: {'delta_cost': ..., 'delta_daly': ..., 'icer': ...}
#         """
#         df_self = self.to_df()
#         df_other = other_analyzer.to_df()
    
#         # Align on uids
#         df_self = df_self.set_index('uid')
#         df_other = df_other.set_index('uid')
#         df_common = df_self.join(df_other, lsuffix='_intv', rsuffix='_base', how='inner')
    
#         # Add YLL if available in future
#         df_common['total_daly_intv'] = df_common['total_yld_intv']
#         df_common['total_daly_base'] = df_common['total_yld_base']
    
#         # Aggregate
#         delta_cost = df_common['total_cost_intv'].sum() - df_common['total_cost_base'].sum()
#         delta_daly = df_common['total_daly_base'].sum() - df_common['total_daly_intv'].sum()  # DALY averted
    
#         icer = delta_cost / delta_daly if delta_daly != 0 else np.inf
    
#         return {
#             'delta_cost': delta_cost,
#             'delta_daly': delta_daly,
#             'icer': icer,
#         }
    

#     def to_df(self):
#         return self.detailed_outputs    