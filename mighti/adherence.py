"""
CASM Adherence Connector for MIGHTI
Applies adherence penalties based on CASM conditions (depression, AUD, etc.).
Adherence affects *all* interventions that expose a `rel_effect` attribute.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ["CASMAdherenceConnector", "CASM_REL_FACTORS_PHARMA", "CASM_REL_FACTORS_LIFESTYLE"]

# ----------------------------------------------------------------------
# Odds ratios from Table S2 → converted to relative adherence (1/OR)
# ----------------------------------------------------------------------
CASM_REL_FACTORS_PHARMA = {
    "AlcoholUseDisorder":      1 / 1.41,
    "MajorDepressiveDisorder": 1 / 2.21,
    "AnxietyDisorder":         1 / 2.04,
    "ChronicPain":             1 / 1.34,
    "TobaccoUseDisorder":      1 / 1.18,
    "OpioidUseDisorder":       1 / 1.18,
    "StimulantUseDisorder":    1 / 1.81,
}

CASM_REL_FACTORS_LIFESTYLE = {
    "AlcoholUseDisorder":      0.90,
    "MajorDepressiveDisorder": 0.85,
    "AnxietyDisorder":         0.88,
    "ChronicPain":             0.90,
    "TobaccoUseDisorder":      0.95,
    "OpioidUseDisorder":       0.95,
    "StimulantUseDisorder":    0.90,
}


class CASMAdherenceConnector(ss.Module):
    """
    Applies CASM-related adherence reductions to interventions.
    Each intervention must set .casm_sensitivity = 'pharma' or 'lifestyle'.
    """
    def __init__(self,
                 rel_factors_pharma=None,
                 rel_factors_lifestyle=None,
                 label="casm_adherence_connector"):
        super().__init__(label=label)
        self.rel_factors_pharma = rel_factors_pharma or CASM_REL_FACTORS_PHARMA.copy()
        self.rel_factors_lifestyle = rel_factors_lifestyle or CASM_REL_FACTORS_LIFESTYLE.copy()
        self.time = sc.autolist()
        self.mean_adherence = sc.autolist()

    def init_pre(self, sim):
        super().init_pre(sim)
        self.setattribute("sim", sim)
        print(f"[CASMAdherenceConnector] Attached to sim: {sim.label}")

    def step(self):
        sim = self.sim
        ppl = sim.people
        n = len(ppl)

        # Build CASM condition mask for all agents
        affected = {}
        for cond, rel in self.rel_factors_pharma.items():  # assume same keys for both
            for dname, dobj in getattr(sim, "diseases", {}).items():
                if cond.lower() in dname.lower() and hasattr(dobj, "affected"):
                    affected[cond] = dobj.affected

        # Loop over interventions
        for intv in sim.interventions:
            sensitivity = getattr(intv, "casm_sensitivity", None)
            if sensitivity not in ["pharma", "lifestyle"]:
                continue  # skip non-CASM-relevant interventions

            adherence = np.ones(n)
            rel_factors = (self.rel_factors_pharma if sensitivity == "pharma"
                           else self.rel_factors_lifestyle)

            for cond, rel in rel_factors.items():
                if cond in affected:
                    adherence[affected[cond]] *= rel

            adherence = np.clip(adherence, 0.0, 1.0)

            if hasattr(intv, "rel_effect"):
                if np.isscalar(intv.rel_effect):
                    intv.rel_effect *= adherence.mean()
                else:
                    intv.rel_effect *= adherence

            self.mean_adherence.append(adherence.mean())
            self.time.append(sim.t)

            print(f"[CASMAdherenceConnector] {sim.t.year}: {intv.label} adherence = {adherence.mean():.3f} ({sensitivity})")

            