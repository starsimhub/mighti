"""
Defines interdependencies and risk modifiers between diseases and conditions.
"""


import pandas as pd
import starsim as ss
import sciris as sc
from collections import defaultdict
import numpy as np


def _ensure_interaction_reset_store(sim):
    """Per-timestep store for resetting destination susceptibilities once."""
    ti = sim.ti
    store = getattr(sim, "_interaction_reset", None)
    if store is None or store.get("ti") != ti:
        store = {"ti": ti, "done": set()}
        setattr(sim, "_interaction_reset", store)
    return store


def _ensure_base_rel_sus_store(sim):
    """Persistent store of baseline rel_sus arrays (captured at init)."""
    base = getattr(sim, "_interaction_base_rel_sus", None)
    if base is None:
        base = {}
        setattr(sim, "_interaction_base_rel_sus", base)
    return base


def _register_base_rel_sus(sim, dest_key, dest_mod):
    """
    Capture baseline rel_sus for a destination module once (at init).
    For most modules this is all ones.
    """
    dest_key = str(dest_key).strip().lower()
    base = _ensure_base_rel_sus_store(sim)
    if dest_key in base:
        return
    try:
        arr = np.asarray(dest_mod.rel_sus).copy()
        base[dest_key] = arr
    except Exception:
        # Fallback: reset will use ones if needed
        base[dest_key] = None


def _reset_rel_sus_to_base(sim, dest_key, dest_mod):
    """Reset dest_mod.rel_sus to baseline, once per timestep per dest_key."""
    dest_key = str(dest_key).strip().lower()
    store = _ensure_interaction_reset_store(sim)
    if dest_key in store["done"]:
        return

    base = _ensure_base_rel_sus_store(sim).get(dest_key, None)
    try:
        if base is None:
            dest_mod.rel_sus[:] = 1.0
        else:
            dest_mod.rel_sus[:] = base
    except Exception:
        pass

    store["done"].add(dest_key)


class NCDHIVConnector(ss.Connector):
    """
    Connector to model interaction between HIV and NCDs by adjusting susceptibility.

    This connector increases the susceptibility to specified NCDs among HIV-infected agents
    by a user-specified relative susceptibility factor. It also tracks and optionally plots
    the dynamics of NCD prevalence, HIV prevalence, and susceptibility over time.

    Attributes:
        rel_sus_dict (dict): Dictionary mapping NCD names (str) to relative susceptibility values (float).
        time (sc.autolist): List of simulation times for plotting.
        rel_sus (defaultdict): Time-series of mean relative susceptibility for each NCD.
        ncd_prev (defaultdict): Time-series of NCD prevalence values.
        hiv_prev (sc.autolist): Time-series of HIV prevalence values.
    """

    def __init__(self, rel_sus_dict, pars=None, **kwargs):

        super().__init__(label='NCD-HIV')
        self.rel_sus_dict = rel_sus_dict
        self.update_pars(pars, **kwargs)
        
        self.time = sc.autolist()
        self.rel_sus = defaultdict(sc.autolist)
        self.ncd_prev = defaultdict(sc.autolist)
        self.hiv_prev = sc.autolist()

    def initialize(self, sim):
        """Capture baseline rel_sus arrays for destination NCDs."""
        try:
            super().initialize(sim)
        except Exception:
            self.sim = sim

        for ncd in self.rel_sus_dict.keys():
            ncd_obj = sim.diseases.get(str(ncd).strip().lower(), None)
            if ncd_obj is not None and hasattr(ncd_obj, "rel_sus"):
                _register_base_rel_sus(sim, str(ncd).strip().lower(), ncd_obj)
        
    def step(self):

        hiv = self.sim.diseases.hiv
        
        for ncd, rel_sus_val in self.rel_sus_dict.items():
            ncd_obj = self.sim.diseases.get(ncd.lower(), None)
            if ncd_obj is not None:
                dest_key = str(ncd).strip().lower()
                _reset_rel_sus_to_base(self.sim, dest_key, ncd_obj)

                # Historical behavior: assignment for HIV → NCD (not multiplicative)
                try:
                    ncd_obj.rel_sus[hiv.infected.uids] = float(rel_sus_val)
                except Exception:
                    pass

                self.rel_sus[ncd].append(ncd_obj.rel_sus.mean())
                self.ncd_prev[ncd].append(ncd_obj.results.prevalence[self.sim.ti])
                
        self.time.append(self.sim.t)
        self.hiv_prev.append(hiv.results.prevalence[self.sim.ti])
        return
    
    def plot(self):

        # Local import to avoid pulling matplotlib on `import mighti`
        import matplotlib.pyplot as plt

        sc.options(dpi=200)
        fig, ax = plt.subplots(len(self.rel_sus_dict), 1, figsize=(10, 8))
        
        if len(self.rel_sus_dict) == 1:
            ax = [ax]
        
        for i, ncd in enumerate(self.rel_sus_dict.keys()):
            ax[i].plot(self.time, self.rel_sus[ncd], label=f'{ncd} rel_sus')
            ax[i].plot(self.time, self.ncd_prev[ncd], label=f'{ncd} prevalence')
            ax[i].plot(self.time, self.hiv_prev, label='HIV prevalence')
            ax[i].legend()
            ax[i].set_title(f'{self.sim.label} - {ncd}')
        
        plt.tight_layout()
        plt.show()
        return fig


def read_interactions(datafile=None):
    """
    Reads interaction data from a CSV file.
    Automatically creates the Connectors based on relative risk.
    """
    if datafile is None:
        datafile = 'rel_sus.csv'
    df = pd.read_csv(datafile, index_col=0)

    rel_sus = defaultdict(dict)

    for condition1 in df.index:
        for condition2 in df.columns:
            if condition1 != condition2:
                value = df.at[condition1, condition2]
                if not pd.isna(value):
                    rel_sus[condition1][condition2] = value

    return rel_sus


def create_connectors(rel_sus):
    connectors = []
    for condition1, interactions in rel_sus.items():
        for condition2, rel_sus_val in interactions.items():
            connector = create_dynamic_connector(condition1, condition2, rel_sus_val)
            connectors.append(connector)
    return connectors


def create_dynamic_connector(condition1, condition2, rel_sus_val):
    """
    Create a Connector implementing condition1 → condition2 susceptibility effects.

    - For non-HIV condition2, modifies condition2.rel_sus among those at risk for condition2.
    - For condition2 == HIV, modifies hiv.rel_sus among HIV-susceptible individuals.

    Notes
    -----
    Multiple connectors can target the same destination condition. To ensure effects do not
    "stick" after remission and to allow multiplicative stacking, each destination rel_sus
    array is reset to baseline once per timestep (per destination) before applying multipliers.
    """
    condition1_key = str(condition1).strip().lower()
    condition2_key = str(condition2).strip().lower()
    label = f"{condition1}-{condition2}"

    class DynamicConnector(ss.Connector):
        def __init__(self, pars=None, **kwargs):
            super().__init__(label=label)
            self.condition1 = condition1_key
            self.condition2 = condition2_key
            self.define_pars(rel_sus=float(rel_sus_val))
            self.update_pars(pars, **kwargs)

            # optional tracking (used only if caller wants to plot)
            self.time = sc.autolist()
            self.rel_sus = sc.autolist()
            self.condition1_prev = sc.autolist()
            self.condition2_prev = sc.autolist()

        def initialize(self, sim):
            """Capture baseline rel_sus for the destination module once."""
            try:
                super().initialize(sim)
            except Exception:
                self.sim = sim

            dst_is_hiv = (self.condition2 == "hiv")
            dst = getattr(sim.diseases, "hiv", None) if dst_is_hiv else sim.diseases.get(self.condition2, None)
            if dst is not None and hasattr(dst, "rel_sus"):
                _register_base_rel_sus(sim, self.condition2, dst)

        def _get_active_uids(self, mod):
            """Return UIDs for the active state of a module (infected or affected)."""
            if mod is None:
                return ss.uids([])
            if hasattr(mod, "infected"):
                try:
                    return mod.infected.uids
                except Exception:
                    pass
            if hasattr(mod, "affected"):
                try:
                    return mod.affected.uids
                except Exception:
                    pass
            return ss.uids([])

        def _get_at_risk_uids(self, mod, *, is_hiv=False):
            """Return UIDs at risk of acquiring condition2."""
            if mod is None:
                return ss.uids([])
            if is_hiv:
                # For HIV, at-risk is "susceptible" (not infected)
                if hasattr(mod, "susceptible"):
                    try:
                        return mod.susceptible.uids
                    except Exception:
                        pass
                if hasattr(mod, "infected"):
                    try:
                        return (~mod.infected).uids
                    except Exception:
                        pass
                return ss.uids([])

            # For non-HIV modules, prefer explicit susceptible, else those not affected/infected
            if hasattr(mod, "susceptible"):
                try:
                    return mod.susceptible.uids
                except Exception:
                    pass
            if hasattr(mod, "affected"):
                try:
                    return (~mod.affected).uids
                except Exception:
                    pass
            if hasattr(mod, "infected"):
                try:
                    return (~mod.infected).uids
                except Exception:
                    pass
            return ss.uids([])

        def step(self):
            sim = self.sim

            src = sim.diseases.get(self.condition1, None)
            dst_is_hiv = (self.condition2 == "hiv")
            dst = getattr(sim.diseases, "hiv", None) if dst_is_hiv else sim.diseases.get(self.condition2, None)

            if src is None or dst is None:
                return

            # Reset destination susceptibility to baseline once per timestep
            _reset_rel_sus_to_base(sim, self.condition2, dst)

            src_active = self._get_active_uids(src)
            dst_at_risk = self._get_at_risk_uids(dst, is_hiv=dst_is_hiv)
            if len(src_active) == 0 or len(dst_at_risk) == 0:
                return

            # Apply multiplier to intersection: active in condition1 and at-risk for condition2
            inter = np.intersect1d(
                np.asarray(src_active, dtype=int),
                np.asarray(dst_at_risk, dtype=int),
                assume_unique=False,
            )
            if inter.size == 0:
                return

            factor = float(getattr(self.pars, "rel_sus", rel_sus_val))
            if hasattr(dst, "rel_sus"):
                dst.rel_sus[ss.uids(inter)] *= factor
                self.rel_sus.append(float(np.mean(dst.rel_sus[ss.uids(inter)])))

            # Optional tracking for plots
            self.time.append(sim.t)
            try:
                if hasattr(src, "results") and hasattr(src.results, "prevalence"):
                    self.condition1_prev.append(src.results.prevalence[sim.ti])
            except Exception:
                pass
            try:
                if hasattr(dst, "results") and hasattr(dst.results, "prevalence"):
                    self.condition2_prev.append(dst.results.prevalence[sim.ti])
            except Exception:
                pass

        def plot(self):
            return plot_function(self)

    DynamicConnector.__name__ = f"{condition1}_{condition2}_Connector"
    return DynamicConnector()


def step_function(self, condition1, condition2, rel_sus_val):
    """
    Deprecated legacy helper (kept for backward compatibility).
    Prefer the class-based dynamic connector created by `create_dynamic_connector()`.
    """
    return


def plot_function(self):
    sc.options(dpi=200)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for key in ['rel_sus', 'condition1_prev', 'condition2_prev']:
        plt.plot(self.time, self[key], label=key)
    plt.legend()
    plt.title(self.sim.label)
    plt.show()
    return fig
