"""
Major Depressive Disorder (MDD) as a remitting disease with
housing-dependent recovery and a DepressionCare intervention.
"""

import numpy as np
import starsim as ss
import pandas as pd
import logging
from mighti.diseases.base_disease import RemittingDisease
from mighti.rng import get_rng

logger = logging.getLogger(__name__)


# =====================================================================
# Disease definition
# =====================================================================
class MajorDepressiveDisorder(RemittingDisease):
    """
    Major Depressive Disorder (MDD) — modeled as a remitting disease.

    Key features
    ------------
    • Uses per-step remission probability (pars.remission_rate)
    • Boosts remission among housed individuals (housing_protect)
    • Compatible with external interventions such as DepressionCare
    """

    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = "MajorDepressiveDisorder"
        super().__init__(csv_path, pars=None, **kwargs)

        # Add subclass-specific parameters
        self.define_pars(
            remission_mult=1.0,   # global multiplier on remission rate
            housing_protect=0.5,  # +50% remission boost for housed
        )

        if pars:
            self.update_pars(pars)

    # ------------------------------------------------------------------
    def _get_housing_mask(self, uids=None):
        """Return a boolean mask showing who is housed."""
        ppl = self.sim.people
        n = len(ppl) if uids is None else len(uids)

        if hasattr(ppl, "has_housing"):
            arr = ppl.has_housing
        elif hasattr(ppl, "neighbourhood_situation"):
            arr = ppl.neighbourhood_situation
        else:
            return np.zeros(n, dtype=bool)

        return arr if uids is None else arr[uids]

    # ------------------------------------------------------------------
    def step_state(self):
        """
        Apply remission each timestep, with housing-dependent boost.
        """
        ti = self.ti
        affected_uids = self.affected.uids
        if len(affected_uids) == 0:
            return

        # Base remission rate × multiplier
        base = float(self.pars.remission_rate) * float(self.pars.remission_mult)
        base = max(base, 0.0)

        # Housing boost
        housed_mask = self._get_housing_mask(affected_uids)
        boost = float(self.pars.housing_protect)
        p = np.full(len(affected_uids), base, dtype=float)
        p[housed_mask] = base * (1.0 + boost)
        p = np.clip(p, 0.0, 0.999999)

        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        recovered_mask = rng.random(len(affected_uids)) < p
        recovered = affected_uids[recovered_mask]

        if len(recovered):
            self.affected[recovered] = False
            self.reversed[recovered] = True
            self.ti_reversed[recovered] = ti

        # Immediately move reversed → susceptible
        recovered2 = (self.reversed & (self.ti_reversed <= ti)).uids
        if len(recovered2):
            self.reversed[recovered2] = False
            self.susceptible[recovered2] = True

    # ------------------------------------------------------------------
    def step(self):
        """
        Step progression, deaths, and housing-aware remission.
        """
        new_cases = super().step()  # acquisition, deaths, results
        self.step_state()            # apply remission logic

        # Update prevalence metrics
        ppl_n = len(self.sim.people)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / ppl_n
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / ppl_n
        return new_cases


# =====================================================================
# Intervention definition
# =====================================================================
class DepressionCare(ss.treat_num):
    """
    Depression treatment intervention (parallel to T2D_ReduceMortalityTx).

    Effects:
        1. Boosts remission among treated individuals.
        2. Optionally improves ART adherence via CASM connector.
    """

    def __init__(self, *args, product=None, prob=1.0,
                 remission_boost=1.5, adherence_boost=1.1,
                 eligibility=None, **kwargs):

        self.disease = "majordepressivedisorder"
        self.remission_boost = remission_boost
        self.adherence_boost = adherence_boost

        if product is not None and hasattr(product, "df"):
            df = product.df
            if "disease" in df.columns:
                df = df[df["disease"].str.lower() == self.disease]
            product.df = df  # modify in place, don’t wrap again
        super().__init__(*args, product=product, prob=prob,
                         eligibility=eligibility, **kwargs)


    def initialize(self, sim):
        super().initialize(sim)
        if self.eligibility is None:
            if not hasattr(sim.diseases, self.disease):
                raise ValueError(f"[{self.label}] Disease '{self.disease}' not found.")
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids

        # Debug logging only (avoid stdout noise in production)
        dep = sim.diseases[self.disease]
        logger.debug(
            "[DepressionCare] %s linked to '%s' (affected at start=%s, prob=%s; eligibility_set=%s)",
            getattr(self, "label", "DepressionCare"),
            self.disease,
            int(dep.affected.sum()),
            self.prob,
            self.eligibility is not None,
        )

    def step(self):
        """Treat eligible individuals and apply remission/adherence effects."""
        # 1) bookkeeping
        cur_year = float(self.sim.now)
        dep = self.sim.diseases[self.disease]

        # 2) ensure eligibility function exists
        if self.eligibility is None:
            if not hasattr(self.sim.diseases, self.disease):
                raise ValueError(f"[{self.label}] Disease '{self.disease}' not found in sim.diseases.")
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids
            logger.debug("[DepressionCare] %s eligibility auto-set at year %.1f", self.label, cur_year)

        # 3) who’s eligible this step?
        eligible = self.eligibility(self.sim)
        n_eligible = len(eligible)
        logger.debug("[DepressionCare] %s year %.1f | eligible=%s", self.label, cur_year, n_eligible)

        # 3) apply coverage probability
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        chooser = (rng.random(n_eligible) < self.prob)
        treated = eligible[chooser]
        self.treated_inds = ss.uids(treated)
        logger.debug("[DepressionCare] %s treated=%s (prob=%s)", self.label, len(treated), self.prob)

        # 4) apply effects
        if len(treated):
            try:
                dep.pars.remission_mult = float(self.remission_boost)
                logger.debug(
                    "[DepressionCare] %s boosted remission_mult -> %.3f",
                    self.label,
                    float(dep.pars.remission_mult),
                )
            except Exception as e:
                logger.debug("[DepressionCare] %s remission boost skipped: %s", self.label, e)

            # Try to nudge ART if present
            for intv in self.sim.interventions.values():
                if hasattr(intv, "label") and "ART" in intv.label.upper():
                    if hasattr(intv, "adherence_scale"):
                        intv.adherence_scale *= self.adherence_boost
                        logger.debug(
                            "[DepressionCare] %s ART adherence_scale -> %s",
                            self.label,
                            intv.adherence_scale,
                        )
                    break

        return self.treated_inds    
    