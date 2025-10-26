import starsim as ss
import numpy as np
import pandas as pd

class MicrocostingAnalyzer(ss.Analyzer):
    def __init__(self, unit_costs=None, disability_weights=None,
                 discount_rate=None,
                 discount_rate_costs=0.03, discount_rate_outcomes=0.03,
                 **kwargs):

        if discount_rate is not None:
            discount_rate_costs = discount_rate_outcomes = discount_rate

        self.unit_costs = unit_costs or {}
        self.disability_weights = disability_weights or {}
        self.discount_rate_costs = discount_rate_costs
        self.discount_rate_outcomes = discount_rate_outcomes

        # Remove these to prevent warnings from Starsim
        for k in ['discount_rate', 'discount_rate_costs', 'discount_rate_outcomes']:
            kwargs.pop(k, None)

        super().__init__(**kwargs)
        self.name = 'microcostinganalyzer'
        self.detailed_outputs = None

    def init_results(self):
        super().init_results() 
        self.results = ss.Results(self)  
        return

    def step(self):
        """Dummy step to silence Starsim warnings."""
        pass

    def finalize(self):
        super().finalize()

        ppl = self.sim.people
        n_total = ppl.n_uids      
        n_alive = len(ppl)
        uids_all = np.arange(n_total)
        years = self.sim.t.yearvec
        n_years = len(years)

        print(f"\n Finalizing MicrocostingAnalyzer for {n_alive:,} alive / {n_total:,} total agents across {n_years} years")

        # Initialize arrays at full population size
        total_cost = np.zeros(n_total)
        total_yld  = np.zeros(n_total)
        total_yll  = np.zeros(n_total)
        cost_details, yld_details, yll_details = {}, {}, {}

        # ---------------------------------------------------------------------
        # Event-based costs
        # ---------------------------------------------------------------------
        print("\n Event-based costs:")
        for event, unit_cost in self.unit_costs.items():
            if event == 'art':
                continue
            if hasattr(ppl, f'{event}_count'):
                counts = getattr(ppl, f'{event}_count')
                if isinstance(counts, dict):
                    arr = np.array([counts.get(uid, 0) for uid in uids_all])
                else:
                    arr = np.asarray(counts)
                    if len(arr) < n_total:
                        arr = np.pad(arr, (0, n_total - len(arr)))
                cost = arr * unit_cost / ((1 + self.discount_rate_costs) ** (n_years - 1))
                total_cost += cost
                cost_details[f'{event}_cost'] = cost
                print(f"  • {event}: {cost.sum():,.2f}")

        # ---------------------------------------------------------------------
        # YLDs
        # ---------------------------------------------------------------------
        print("\n YLDs by condition:")

        for cond, weight in self.disability_weights.items():

            # Special logic for HIV (ti_infected-based)
            if cond == 'hiv' and hasattr(ppl, 'hiv') and hasattr(ppl.hiv, 'ti_infected'):
                print(f"  Calculating HIV YLDs dynamically from ti_infected")

                ti_infected_arr = np.full(n_total, np.nan)
                ti_infected_arr[:len(ppl.hiv.ti_infected)] = ppl.hiv.ti_infected[:]  # Fill full-length array

                infected_mask = ~np.isnan(ti_infected_arr)
                ti_infected_clipped = np.clip(ti_infected_arr[infected_mask], 0, len(self.sim.t.yearvec) - 1).astype(int)
                start_years = self.sim.t.yearvec[ti_infected_clipped]
                end_year = self.sim.t.yearvec[-1]
                dur_years = end_year - start_years

                yld = np.zeros(n_total)
                yld[infected_mask] = dur_years * weight / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                yld_details[f'{cond}_yld'] = yld
                total_yld += yld
                print(f"  {cond}: {yld.sum():.2f}")
                continue

            # New: dynamic condition duration lookup
            elif hasattr(self.sim.diseases, cond) and hasattr(self.sim.diseases[cond], 'duration'):
                print(f"  Calculating {cond} YLDs from disease.duration")
                disease = self.sim.diseases[cond]
                durations = disease.duration
                yld = durations * weight / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                if len(yld) < n_total:
                    yld = np.pad(yld, (0, n_total - len(yld)))
                total_yld += yld
                yld_details[f'{cond}_yld'] = yld
                print(f"  {cond}: {yld.sum():.2f}")

            else:
                print(f"  Missing: {cond} or duration attribute not found")
        
        
        # ---------------------------------------------------------------------
        # YLLs
        # ---------------------------------------------------------------------
        print("\n YLLs from ConditionAtDeathAnalyzer:")
        condition_death = self.sim.analyzers.get('condition_at_death_analyzer', None)
        if condition_death and hasattr(condition_death, 'to_df'):
            df_yll = condition_death.to_df()
            n_deaths = len(df_yll)
            print(f"  Number of deaths recorded: {n_deaths:,}")
            if n_deaths:
                yll_array_discounted = df_yll['yll'].to_numpy() / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                discounted_series = pd.Series(yll_array_discounted, index=df_yll['uid'])
                mapped = discounted_series.reindex(uids_all, fill_value=0.0).to_numpy()
                total_yll += mapped
                yll_details['yll'] = mapped
                print(f"  • Total YLLs: {mapped.sum():,.2f}")
        else:
            print(" ConditionAtDeathAnalyzer not found")

        # ---------------------------------------------------------------------
        # ART costs
        # ---------------------------------------------------------------------
        print("\n ART costing:")
        intervention_analyzer = self.sim.analyzers.get('intervention_analyzer', None)
        if intervention_analyzer is None:
            raise ValueError("MicrocostingAnalyzer requires 'intervention_analyzer' in sim.analyzers.")
        if 'art' in self.unit_costs:
            art_df = intervention_analyzer.to_df()
            art_counts = (
                art_df[art_df['received_art']]
                .groupby('uid').size()
                .reindex(uids_all, fill_value=0)
                .to_numpy()
            )
            art_cost = art_counts * self.unit_costs['art'] / ((1 + self.discount_rate_costs) ** (n_years - 1))
            total_cost += art_cost
            cost_details['art_cost'] = art_cost
            print(f"  • ART total cost: {art_cost.sum():,.2f} for {art_counts.sum():,} doses")

        # ---------------------------------------------------------------------
        # Final DataFrame
        # ---------------------------------------------------------------------
        df = pd.DataFrame({
            'uid': uids_all,
            'total_cost': total_cost,
            'total_yld': total_yld,
            'total_yll': total_yll,
            'total_daly': total_yld + total_yll,
        })

        for key, val in {**cost_details, **yld_details, **yll_details}.items():
            df[key] = val

        self.detailed_outputs = df

        print(f"\n Finalized MicrocostingAnalyzer")
        print(f"   → Total cost: ${total_cost.sum():,.2f}")
        print(f"   → Total YLD: {total_yld.sum():,.2f}")
        print(f"   → Total YLL: {total_yll.sum():,.2f}")
        print(f"   → Total DALY: {df['total_daly'].sum():,.2f}")

        print(f"\n Finalized MicrocostingAnalyzer")
        print(f"   → Total cost: ${total_cost.sum():,.2f}")
        print(f"   → Total YLD: {total_yld.sum():,.2f}")
        print(f"   → Total YLL: {total_yll.sum():,.2f}")
        print(f"   → Total DALY: {df['total_daly'].sum():,.2f}")

        # ---------------------------------------------------------------------
        # Store summary results for programmatic access
        # ---------------------------------------------------------------------
        self.results['total_cost'] = total_cost.sum()
        self.results['total_yld'] = total_yld.sum()
        self.results['total_yll'] = total_yll.sum()
        self.results['total_daly'] = (total_yld + total_yll).sum()

        # Optionally store the detailed dataframe too
        self.results['detailed_outputs'] = self.detailed_outputs

        return

    def compute_icer(self, other_analyzer):
        df_self = self.to_df().set_index('uid')
        df_other = other_analyzer.to_df().set_index('uid')
        df_common = df_self.join(df_other, lsuffix='_intv', rsuffix='_base', how='inner')
        delta_cost = df_common['total_cost_intv'].sum() - df_common['total_cost_base'].sum()
        delta_daly = df_common['total_daly_base'].sum() - df_common['total_daly_intv'].sum()
        icer = delta_cost / delta_daly if delta_daly != 0 else np.inf
        return {'delta_cost': delta_cost, 'delta_daly': delta_daly, 'icer': icer}

    def to_df(self):
        return self.detailed_outputs


def summarize_microcosting_results(analyzer):
    """
    Summarize total cost, YLL, YLD, and DALYs from a MicrocostingAnalyzer,
    including per-condition and per-event breakdowns if available.
    """
    if not hasattr(analyzer, 'detailed_outputs') or analyzer.detailed_outputs is None:
        raise ValueError("Analyzer does not contain detailed_outputs. Run the simulation first.")

    df = analyzer.detailed_outputs

    summary = {
        'total_cost': df['total_cost'].sum(),
        'total_yll': df['total_yll'].sum(),
        'total_yld': df['total_yld'].sum(),
        'total_daly': df['total_daly'].sum(),
    }

    # Include all columns that end in _yld or _cost (per-condition/per-event)
    for col in df.columns:
        if col.endswith('_yld') or col.endswith('_cost'):
            summary[col] = df[col].sum()

    return summary   
