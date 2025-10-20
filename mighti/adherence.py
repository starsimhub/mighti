"""
Base Adherence Module for MIGHTI
Links adherence to CASM conditions (e.g., depression, AUD, pain)
and modifies the effectiveness of target interventions accordingly.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ["AdherenceFromDepression","AdherenceFromAlcoholUse","AdherenceFromAnxiety","AdherenceFromChronicPain",
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
            for dname, dobj in sim.diseases.items():
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
                    i.rel_effect *= adherence


class AdherenceFromDepression(BaseAdherenceConnector):
    def __init__(self, target_intervention='ART', label=None):
        super().__init__(target_intervention, {'MajorDepressiveDisorder': 0.45},
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
