"""
Module defining Major Depressive Disorder as a remitting disease model
with housing-dependent recovery and protection effects.
"""

from mighti.diseases.base_disease import RemittingDisease
import starsim as ss
import numpy as np



class MajorDepressiveDisorder(RemittingDisease):
    """
    Major Depressive Disorder (MDD) as a remitting disease.
    - Uses per-step remission probability.
    - Remission is multiplied by `remission_mult`.
    - If housed (via `people.has_housing` or `people.neighbourhood_situation`),
      remission is boosted by `housing_protect` (e.g., 0.5 = +50%).
    """

    def __init__(self, csv_path, pars=None, **kwargs):
        # Name for CSV lookups in RemittingDisease.get_disease_parameters()
        self.disease_name = "MajorDepressiveDisorder"

        # Construct the base class WITHOUT passing pars yet (so we can add new pars)
        super().__init__(csv_path, pars=None, **kwargs)

        # Add subclass-specific parameters (now safe to accept via update_pars)
        self.define_pars(
            remission_mult=1.0,   # global multiplier on remission_rate
            housing_protect=0.5,  # +50% remission for housed by default
        )

        # Now apply caller-provided pars (including remission_mult, housing_protect)
        if pars:
            self.update_pars(pars)

        # IMPORTANT: we will handle remission ourselves (to avoid distribution init issues),
        # so we won't use self.p_remission from the base; no need to redefine it.

    def _get_housing_mask(self, uids=None):
        """
        Return a boolean mask (len = #people OR len(uids)) with 'True' where housed.
        Accepts either `people.has_housing` or `people.neighbourhood_situation`.
        """
        ppl = self.sim.people
        n = len(ppl) if uids is None else len(uids)

        if hasattr(ppl, 'has_housing'):
            arr = ppl.has_housing
        elif hasattr(ppl, 'neighbourhood_situation'):
            arr = ppl.neighbourhood_situation
        else:
            return np.zeros(n, dtype=bool)  # nobody housed if attribute doesn't exist

        return arr if uids is None else arr[uids]

    def step_state(self):
        """
        Override remission logic so we can apply a housing-dependent boost
        without relying on an ss.bernoulli distribution.
        """
        ti = self.ti

        # Recoveries only among currently affected
        affected_uids = self.affected.uids
        if len(affected_uids) == 0:
            return  # nothing to do

        # Base per-step remission (already per-step in your RemittingDisease)
        base = float(self.pars.remission_rate) * float(self.pars.remission_mult)
        if base < 0.0:
            base = 0.0

        # Housing boost
        housed_mask = self._get_housing_mask(affected_uids)
        boost = float(self.pars.housing_protect)
        per_person_p = np.full(len(affected_uids), base, dtype=float)
        per_person_p[housed_mask] = base * (1.0 + boost)

        # Clamp to [0,1) to be safe
        per_person_p = np.clip(per_person_p, 0.0, 0.999999)

        # Draw recoveries
        draws = np.random.rand(len(affected_uids)) < per_person_p
        recovered = affected_uids[draws]

        # Update states — mirror your RemittingDisease logic
        if len(recovered):
            self.affected[recovered] = False
            self.reversed[recovered] = True
            self.ti_reversed[recovered] = ti

        # Move from "reversed" (remission) back to susceptible immediately
        # (keeps your base semantics — if you want a dwell time, add it here)
        recovered2 = (self.reversed & (self.ti_reversed <= ti)).uids
        if len(recovered2):
            self.reversed[recovered2] = False
            self.susceptible[recovered2] = True

        # NOTE: we intentionally do NOT call super().step_state()
        # because the base would apply its own remission draw via self.p_remission,
        # leading to double remission.

    def step(self):
        """
        Keep base dynamics for acquisition and death, but let our step_state()
        handle remission. We call the parent's `step()` to do:
          - new cases (acquisition)
          - deaths
          - results updates (new_cases, new_deaths, prevalence, etc.)
        The parent's step() also writes `results.prevalence` etc., which we want.
        """
        # Do parent acquisition + death + results (this will NOT handle remission)
        new_cases = super().step()

        # Immediately after, run our remission logic (housing-aware)
        self.step_state()

        # Update results again to reflect any remission that happened this step
        # (ensures prevalence in results matches the post-remission state)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

        return new_cases



from starsim.interventions import BaseTreatment


class DepressionCare(ss.treat_num):
    """
    Treats individuals with depression (MajorDepressiveDisorder),
    boosting remission and optionally improving ART adherence.

    Args:
        product (ss.Product): optional treatment product
        prob (float): per-step treatment probability (coverage level)
        max_capacity (int): optional limit on treated agents per step
        eligibility (callable): custom eligibility function
        label (str): intervention label
        remission_boost (float): multiplier on remission rate for treated individuals
        adherence_boost (float): multiplier on ART adherence (CASM link)
        verbose (bool): print summary each step
    """

    def __init__(self, product=None, prob=1.0, max_capacity=None, eligibility=None,
                 label='DepressionCare', remission_boost=1.5, adherence_boost=1.1,
                 verbose=False, **_):
        """
        Starsim 3.x passes kwargs to Timeline; extra fields like remission_boost must
        not be forwarded, so we explicitly exclude them.
        """
        super().__init__(product=product, prob=prob, max_capacity=max_capacity,
                         eligibility=eligibility, label=label)
        self.disease_name = 'MajorDepressiveDisorder'
        self.remission_boost = remission_boost
        self.adherence_boost = adherence_boost
        self.verbose = verbose
        self.treated_inds = []
        self.treated_per_timestep = []  # optional: track number treated per step

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_pre(self, sim):
        """Define eligibility before the simulation starts."""
        super().init_pre(sim)
        if self.eligibility is None:
            dis_key = self.disease_name.lower()
            if dis_key not in sim.diseases:
                raise ValueError(f"[{self.label}] Disease '{self.disease_name}' not found in sim.diseases.")
            self.eligibility = lambda sim: sim.diseases[dis_key].affected.uids

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self):
        """Apply treatment, boost remission, and optionally ART adherence."""
        sim = self.sim
        dep = sim.diseases[self.disease_name.lower()]

        # Select eligible individuals
        eligible = self.eligibility(sim)
        if len(eligible) == 0:
            self.treated_per_timestep.append(0)
            return np.array([], dtype=int)

        # Apply coverage probability
        to_treat = eligible[np.random.rand(len(eligible)) < self.prob]

        # Starsim 3.0.3: apply treatment using BaseTreatment.step()
        treat_inds = BaseTreatment.step(self)
        self.treated_inds = treat_inds
        self.treated_per_timestep.append(len(treat_inds))

        # Apply effects
        if len(treat_inds):
            # 1. Boost remission
            dep.pars.remission_rate[treat_inds] *= self.remission_boost

            # 2. Optionally, modest adherence improvement if ART intervention present
            art_intv = next((i for i in sim.interventions if 'ART' in i.label.upper()), None)
            if art_intv is not None and hasattr(art_intv, 'rel_effect'):
                art_intv.rel_effect[treat_inds] *= self.adherence_boost

            if self.verbose:
                print(f"[{self.label}] Treated {len(treat_inds)} agents at step {sim.t.ti}")

        return treat_inds
    