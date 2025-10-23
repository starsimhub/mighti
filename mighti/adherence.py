"""
Base Adherence Module for MIGHTI
Links adherence to CASM conditions (e.g., depression, AUD, pain)
and modifies the effectiveness of target interventions accordingly.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ["AdherenceDisruptFromDepression","AdherenceFromDepression","AdherenceFromAlcoholUse","AdherenceFromAnxiety","AdherenceFromChronicPain",
           "AdherenceFromTobaccoUse","AdherenceFromOpioidUse","AdherenceFromStimulantUse"]


class BaseAdherenceConnector(ss.Module):
    """
    Connector-module hybrid: adjusts adherence based on CASM conditions
    and logs mean adherence per timestep.
    """

    def __init__(self, target_intervention='ART', rel_factors=None, label=None):
        label = label or f"adherence_{target_intervention.lower()}"
        super().__init__(label=label)

        # internal attributes
        self.target_intervention = target_intervention
        self.rel_factors = rel_factors or {}
        self.time = sc.autolist()
        self.mean_adherence = sc.autolist()

    # ------------------------------------------------------------------
    # Initialization hook — ensures it's part of sim.modules
    # ------------------------------------------------------------------
    def init_pre(self, sim):
        super().init_pre(sim)
        self.setattribute('sim', sim)
        print(f"[DEBUG INIT] Attached {self.label} to sim '{sim.label}'")

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------
    def step(self):
        sim = self.sim
        if sim is None or sim.people is None:
            return

        ppl = sim.people
        n = len(ppl)
        adherence = np.ones(n)

        # --- Apply CASM effects ---
        for cond_name, rel_val in self.rel_factors.items():
            cond_key = cond_name.lower()
            for dname, dobj in getattr(sim, "diseases", {}).items() if isinstance(sim.diseases, dict) else enumerate(sim.diseases):
                if cond_key in dname.lower() and hasattr(dobj, "affected"):
                    affected = dobj.affected
                    adherence[affected] *= rel_val

        adherence = np.clip(adherence, 0.0, 1.0)
        self.mean_adherence.append(adherence.mean())
        self.time.append(sim.t)
        print(f"[DEBUG STEP] {sim.t.year}: mean adherence = {adherence.mean():.3f}")

        # --- Link to target intervention ---
        for i in sim.interventions:
            if getattr(i, "label", "").lower() == self.target_intervention.lower():
                if hasattr(i, "rel_effect"):
                    if np.isscalar(i.rel_effect):
                        i.rel_effect *= adherence.mean()
                    else:
                        i.rel_effect *= adherence


class AdherenceDisruptFromDepression(ss.Connector):
    """
    Depression → ART disruption connector for sti.HIV.
    Handles only adherence effects (no remission or adherence boosts from care).
    """

    def __init__(self,
                 p_drop_dep_per_year=0.25,   # slightly stronger to reveal effect
                 p_restart_rec_per_year=0.05,
                 label="AdherenceDisruptFromDepression"):
        super().__init__(label=label)
        self.p_drop_dep_per_year = p_drop_dep_per_year
        self.p_restart_rec_per_year = p_restart_rec_per_year
        self._initialized = False
        self._dt_local = 1.0
        self.p_drop_step = None
        self.p_restart_step = None

    def initialize(self, sim):
        """Standard Starsim init (safe to re-call in parallel)."""
        dt = getattr(getattr(sim, "pars", None), "dt", None)
        if dt is None or not isinstance(dt, (float, int)):
            self._dt_local = 1.0
            print("[AdhDisrupt] Warning: sim.pars.dt was None → defaulting to 1.0 year step")
        else:
            self._dt_local = float(dt)

        self.p_drop_step = 1 - (1 - self.p_drop_dep_per_year) ** self._dt_local
        self.p_restart_step = 1 - (1 - self.p_restart_rec_per_year) ** self._dt_local
        self._initialized = True
        print(f"[AdhDisrupt] initialized | drop={self.p_drop_step:.3f}, restart={self.p_restart_step:.3f}, dt={self._dt_local}")
        

    def _ensure_init(self):
        if not self._initialized:
            sim = getattr(self, "sim", None)
            if sim is not None:
                self.initialize(sim)
            else:
                self.p_drop_step = 1 - (1 - self.p_drop_dep_per_year)
                self.p_restart_step = 1 - (1 - self.p_restart_rec_per_year)
                self._initialized = True

    # ------------------------------------------------------------------
    def step(self):
        """Drop ART for depressed; restart for recovered."""
        self._ensure_init()
        st = self.sim.people.states
        if "hiv.on_art" not in st or "majordepressivedisorder.affected" not in st:
            return

        on_art = np.asarray(st["hiv.on_art"], dtype=bool)
        dep    = np.asarray(st["majordepressivedisorder.affected"], dtype=bool)
        rec    = np.asarray(st.get("majordepressivedisorder.reversed",
                                   np.zeros_like(dep, bool)), dtype=bool)

        hiv = getattr(self.sim.diseases, "hiv", None)
        if hiv is None:
            return

        N = len(dep)
        uids_all = np.arange(N)

        # --- ART dropouts among depressed ---
        drop_mask = on_art & dep & (np.random.rand(N) < self.p_drop_step)
        drop_ids = uids_all[drop_mask]
        if len(drop_ids):
            # Filter to only those who have valid ART start times
            ti_art = getattr(hiv, "ti_art", None)
            if ti_art is not None:
                valid = ~np.isnan(ti_art[drop_ids]) & (ti_art[drop_ids] >= 0)
                drop_ids = drop_ids[valid]

            if len(drop_ids):
                hiv.stop_art(drop_ids)
                print(f"[AdhDisrupt] Dropped ART for {len(drop_ids)} valid depressed individuals")
            else:
                print("[AdhDisrupt] No valid ART individuals to drop this step (all invalid)")

        # --- ART restarts among recovered ---
        restart_mask = (~on_art) & rec & (~dep) & (np.random.rand(N) < self.p_restart_step)
        restart_ids = uids_all[restart_mask]
        if len(restart_ids):
            hiv.start_art(restart_ids)
            print(f"[AdhDisrupt] Restarted ART for {len(restart_ids)} recovered individuals")


class AdherenceFromDepression(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'MajorDepressiveDisorder': 0.8},
                         label or 'adherence_from_depression')


class AdherenceFromAlcoholUse(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'AlcoholUseDisorder': 0.7},
                         label or 'adherence_from_aud')


class AdherenceFromAnxiety(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'AnxietyDisorder': 0.75},
                         label or 'adherence_from_anxiety')


class AdherenceFromChronicPain(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'ChronicPain': 0.6},
                         label or 'adherence_from_chronicpain')


class AdherenceFromTobaccoUse(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'TobaccoUseDisorder': 0.85},
                         label or 'adherence_from_tobacco')


class AdherenceFromOpioidUse(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'OpioidUseDisorder': 0.65},
                         label or 'adherence_from_opioid')


class AdherenceFromStimulantUse(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'StimulantUseDisorder': 0.6},
                         label or 'adherence_from_stimulant')
        self.rel_factors = {'StimulantUseDisorder': 0.6}
