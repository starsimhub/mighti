"""
Unified Adherence System for MIGHTI (opt-in)
-------------------------------------------

This module is ported from the functionality repository for compatibility with
the "how-to" paper code paths. It implements an HRMM-style adherence model:

- Baseline adherence probability under pharmacotherapy: p0 = 0.62
- Each active CASM condition multiplies the *odds of non-adherence* by an OR
- ORs multiply across comorbid CASM conditions
- Adherence = 1 - P(non-adherence)

Important
---------
This module is intentionally **not** imported into the top-level namespace to
avoid changing default behavior. Use explicitly:

    import mighti.adherence_unified as aud
"""

from __future__ import annotations

import logging
import numpy as np
import sciris as sc
import starsim as ss

from mighti.rng import get_rng

logger = logging.getLogger(__name__)

__all__ = [
    "AdherenceEngine",
    "ARTAdherenceDisruptor",
    "InterventionAdherenceDisruptor",
    "AdherenceFromDepression",
    "BASELINE_ADHERENCE_PHARMACOTHERAPY",
    "CASM_NONADHERENCE_OR",
]


# ---------------------------------------------------------------------
# Defaults (HRMM-style)
# ---------------------------------------------------------------------
BASELINE_ADHERENCE_PHARMACOTHERAPY = 0.62

# Odds ratio (OR) for NON-adherence in presence of CASM conditions
CASM_NONADHERENCE_OR = {
    "AlcoholUseDisorder": 1.41,
    "MajorDepressiveDisorder": 2.21,
    "AnxietyDisorder": 2.04,
    "ChronicPain": 1.34,
    "TobaccoUse": 1.18,
    "OpioidUseDisorder": 1.18,
    "StimulantUseDisorder": 1.81,
}


class AdherenceEngine(ss.Module):
    """Compute per-agent adherence and write into `people.states['adherence']`."""

    def __init__(
        self,
        baseline_adherence: float = BASELINE_ADHERENCE_PHARMACOTHERAPY,
        casm_nonadherence_or: dict | None = None,
        label: str = "adherence_engine",
    ):
        super().__init__(label=label)
        self.baseline_adherence = float(baseline_adherence)
        self.casm_nonadherence_or = (casm_nonadherence_or or CASM_NONADHERENCE_OR).copy()

    @staticmethod
    def _odds(p: float) -> float:
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        return p / (1.0 - p)

    @staticmethod
    def _p_from_odds(o: np.ndarray) -> np.ndarray:
        o = np.asarray(o, dtype=float)
        return o / (1.0 + o)

    def init_pre(self, sim):
        super().init_pre(sim)
        st = sim.people.states
        if "adherence" not in st:
            try:
                arr = ss.FloatArr("adherence", default=1.0)
                sim.people.states.append(arr, overwrite=False)
                st["adherence"][:] = 1.0
            except Exception:
                st["adherence"] = np.ones(len(sim.people), dtype=float)

    def step(self):
        ppl = self.sim.people
        st = ppl.states
        n = len(ppl)

        p0 = float(np.clip(self.baseline_adherence, 1e-12, 1.0 - 1e-12))
        odds_nonadh_0 = self._odds(1.0 - p0)
        mult = np.ones(n, dtype=float)

        for cond, or_nonadh in self.casm_nonadherence_or.items():
            key = f"{cond.lower()}.affected"
            if key in st:
                affected = np.asarray(st[key], dtype=bool)
                if len(affected) != n:
                    affected = np.resize(affected, n)
                mult[affected] *= float(or_nonadh)

        odds_nonadh = odds_nonadh_0 * mult
        p_nonadh = self._p_from_odds(odds_nonadh)
        adherence = 1.0 - p_nonadh

        try:
            st["adherence"][:] = np.clip(adherence, 0.0, 1.0)
        except Exception:
            st["adherence"] = np.clip(adherence, 0.0, 1.0)


class ARTAdherenceDisruptor(ss.Connector):
    """
    Induces ART dropout as a function of adherence:

        dropout_probability_i = base_dropout * (1 - adherence_i)
    """

    def __init__(
        self,
        base_dropout=0.10,
        base_dropout_noaud=0.001,
        allow_reinitiation_after_remission=True,
        label="adherence_art_dropout",
    ):
        super().__init__(label=label)
        self.base_dropout = float(base_dropout)
        self.base_dropout_noaud = float(base_dropout_noaud)
        self.allow_reinitiation_after_remission = bool(allow_reinitiation_after_remission)

    def step(self):
        sim = self.sim
        st = sim.people.states
        hiv = getattr(sim.diseases, "hiv", None)
        if hiv is None:
            return

        adher = np.asarray(st.get("adherence", np.ones(len(sim.people))), dtype=float)
        on_art = np.asarray(st.get("hiv.on_art", []), dtype=bool)
        infected = np.asarray(st.get("hiv.infected", np.zeros_like(on_art, bool)), dtype=bool)
        alive = ~np.asarray(st.get("dead", np.zeros_like(on_art, bool)), dtype=bool)

        if len(adher) != len(on_art):
            adher = np.resize(adher, len(on_art))

        aud_key = "alcoholusedisorder.affected"
        if aud_key in st:
            aud = np.asarray(st.get(aud_key, np.zeros_like(on_art, bool)), dtype=bool)
            if len(aud) != len(on_art):
                aud = np.resize(aud, len(on_art))
        else:
            aud = np.zeros_like(on_art, dtype=bool)

        one_minus = 1.0 - adher
        drop_p = np.where(
            aud,
            self.base_dropout * one_minus,
            self.base_dropout_noaud * one_minus,
        )
        drop_p = np.clip(drop_p, 0.0, 1.0)

        eligible = on_art & infected & alive
        if not np.any(eligible):
            return

        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")
        rand = rng.random(len(on_art))
        drop_ids = np.where(eligible & (rand < drop_p))[0]
        if drop_ids.size:
            try:
                hiv.stop_art(ss.uids(drop_ids))
            except Exception:
                pass


class InterventionAdherenceDisruptor(ss.Module):
    """
    Scales intervention effectiveness based on population-level adherence.
    """

    def __init__(self, scale_art_efficacy=False, label="intervention_adherence_disruptor"):
        super().__init__(label=label)
        self.scale_art_efficacy = bool(scale_art_efficacy)
        self._baseline_art_efficacy = None
        self._baseline_rel_effects = {}

    def init_post(self):
        super().init_post()
        sim = self.sim
        if self.scale_art_efficacy:
            hiv = getattr(sim.diseases, "hiv", None)
            if hiv is not None and hasattr(hiv, "pars") and hasattr(hiv.pars, "art_efficacy"):
                try:
                    self._baseline_art_efficacy = float(hiv.pars.art_efficacy)
                except Exception:
                    self._baseline_art_efficacy = None

        intvs = sim.interventions.values() if isinstance(getattr(sim, "interventions", None), dict) else getattr(sim, "interventions", [])
        for intv in intvs:
            if hasattr(intv, "rel_effect"):
                try:
                    self._baseline_rel_effects[intv.label] = sc.dcp(intv.rel_effect)
                except Exception:
                    self._baseline_rel_effects[intv.label] = intv.rel_effect

    def step(self):
        sim = self.sim
        st = sim.people.states
        adher = np.asarray(st.get("adherence", np.ones(len(sim.people))), dtype=float)
        mean_adher = float(np.nanmean(adher)) if len(adher) else 1.0

        if self.scale_art_efficacy and self._baseline_art_efficacy is not None:
            hiv = getattr(sim.diseases, "hiv", None)
            if hiv is not None and hasattr(hiv, "pars") and hasattr(hiv.pars, "art_efficacy"):
                try:
                    hiv.pars.art_efficacy = float(self._baseline_art_efficacy) * mean_adher
                except Exception:
                    pass

        intvs = sim.interventions.values() if isinstance(getattr(sim, "interventions", None), dict) else getattr(sim, "interventions", [])
        for intv in intvs:
            if not hasattr(intv, "rel_effect"):
                continue
            if intv.label not in self._baseline_rel_effects:
                continue
            baseline = self._baseline_rel_effects[intv.label]
            try:
                intv.rel_effect = baseline * mean_adher
            except Exception:
                pass


class AdherenceFromDepression(ss.Connector):
    """
    Minimal compatibility connector: map depression → lower adherence.

    This is intentionally lightweight. Prefer using `AdherenceEngine`.
    """

    def __init__(self, multiplier=0.8, label="adherence_from_depression"):
        super().__init__(label=label)
        self.multiplier = float(multiplier)

    def step(self):
        sim = self.sim
        st = sim.people.states
        key = "majordepressivedisorder.affected"
        if "adherence" not in st or key not in st:
            return
        affected = np.asarray(st[key], dtype=bool)
        st["adherence"][affected] *= self.multiplier
        st["adherence"][:] = np.clip(st["adherence"], 0.0, 1.0)

