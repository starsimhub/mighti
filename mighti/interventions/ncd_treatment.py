"""Reusable chronic-care treatment that preserves the underlying condition."""

from __future__ import annotations

import numpy as np
import starsim as ss

from mighti.analyzers.disability_weights import canonical_dw_key, resolve_disease_module

__all__ = [
    "ELIGIBILITY_MODES",
    "ChronicCareTx",
    "ReduceMortalityTx",
    "chronic_care_eligible_mask",
    "make_ncd_treatment",
]

ELIGIBILITY_MODES = ("affected", "all_affected")


def _raw_bool(state, n: int) -> np.ndarray:
    raw = getattr(state, "raw", None)
    if raw is not None:
        arr = np.asarray(raw[:n], dtype=bool)
        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size))
        return arr
    arr = np.asarray(state, dtype=bool)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size))
    return arr[:n]


def chronic_care_eligible_mask(sim, disease, mode: str = "affected") -> np.ndarray:
    """Return affected people for the built-in generic eligibility modes."""
    n = int(sim.people.uid.len_used)
    affected = _raw_bool(disease.affected, n)
    mode = (mode or "affected").lower()
    if mode in ELIGIBILITY_MODES:
        return affected
    raise ValueError(f"Unknown eligibility mode {mode!r}; choose from {ELIGIBILITY_MODES}")


class ChronicCareTx(ss.Intervention):
    """Keep eligible people on chronic care without clearing ``affected``.

    Parameters
    ----------
    disease : str
        Canonical or alias disease key (hypertension, type2diabetes, …).
    eligibility : str or callable
        ``affected`` / ``all_affected`` = all with the condition.
        A callable receives ``(sim, disease)`` and returns a boolean mask.
    rel_death_on_treatment : float
        Value assigned to ``disease.rel_death`` while treated. It is not
        multiplied across timesteps.
    initiation_prob_per_step, continuation_prob_per_step : float
        Per-timestep probabilities in [0, 1]. Defaults reproduce deterministic
        full coverage and retention.
    service_claim : str, optional
        Named claim to request from an earlier capacity intervention. If no
        clinic queue is present, care is not gated.
    """

    def __init__(
        self,
        disease,
        eligibility="affected",
        rel_death_on_treatment=0.5,
        rel_death_off_treatment=1.0,
        initiation_prob_per_step=1.0,
        continuation_prob_per_step=1.0,
        service_claim=None,
        rng_seed=1,
        *args,
        **kwargs,
    ):
        self.disease_key = canonical_dw_key(disease)
        kwargs.setdefault("name", f"{self.disease_key}_chroniccaretx")
        super().__init__(*args, **kwargs)
        if callable(eligibility):
            self.eligibility = eligibility
        else:
            mode = (eligibility or "affected").lower()
            if mode not in ELIGIBILITY_MODES:
                raise ValueError(
                    f"Unknown eligibility={eligibility!r}; choose from "
                    f"{ELIGIBILITY_MODES} or supply a callable"
                )
            self.eligibility = mode
        self.rel_death_on_treatment = float(rel_death_on_treatment)
        self.rel_death_off_treatment = float(rel_death_off_treatment)
        self.initiation_prob_per_step = self._validate_probability(
            "initiation_prob_per_step", initiation_prob_per_step
        )
        self.continuation_prob_per_step = self._validate_probability(
            "continuation_prob_per_step", continuation_prob_per_step
        )
        self.service_claim = service_claim
        self.rng_seed = int(rng_seed)
        self._rng = np.random.default_rng(self.rng_seed)
        if not getattr(self, "label", None):
            self.label = f"{self.disease_key}_tx"

    @staticmethod
    def _validate_probability(name: str, value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1; got {value}")
        return value

    def _disease(self):
        disease, resolved = resolve_disease_module(self.sim.diseases, self.disease_key)
        if disease is None:
            raise ValueError(
                f"[{self.label}] Disease {self.disease_key!r} not found in sim.diseases"
            )
        if not hasattr(disease, "on_treatment") or not hasattr(disease, "rel_death"):
            raise ValueError(
                f"[{self.label}] {resolved} needs on_treatment and rel_death states"
            )
        return disease

    def _eligible(self, sim, disease, n: int) -> np.ndarray:
        if callable(self.eligibility):
            mask = np.asarray(self.eligibility(sim, disease), dtype=bool)
            if mask.ndim != 1 or mask.size != n:
                raise ValueError(
                    f"[{self.label}] eligibility callable must return a boolean "
                    f"mask of length {n}; got shape {mask.shape}"
                )
            return mask
        return chronic_care_eligible_mask(sim, disease, self.eligibility)

    def _accepted(self, candidates: np.ndarray, probability: float) -> np.ndarray:
        if probability >= 1.0:
            return candidates.copy()
        if probability <= 0.0:
            return np.zeros_like(candidates)
        accepted = np.zeros_like(candidates)
        uids = np.nonzero(candidates)[0]
        accepted[uids] = self._rng.random(uids.size) < probability
        return accepted

    def step(self):
        sim = self.sim
        disease = self._disease()
        n = int(sim.people.uid.len_used)
        eligible = self._eligible(sim, disease, n)
        on_tx = _raw_bool(disease.on_treatment, n)

        retained = self._accepted(
            on_tx & eligible,
            self.continuation_prob_per_step,
        )
        queue = getattr(sim, "_mighti_clinic_queue", None)
        gate = None
        if queue is not None and self.service_claim is not None:
            gate = queue.served_mask(self.service_claim, n)
            retained &= gate

        drop = on_tx & ~retained

        if drop.any():
            uids = np.nonzero(drop)[0]
            disease.on_treatment[uids] = False
            disease.rel_death[uids] = self.rel_death_off_treatment

        if retained.any():
            uids = np.nonzero(retained)[0]
            disease.rel_death[uids] = self.rel_death_on_treatment

        start = self._accepted(
            eligible & ~on_tx,
            self.initiation_prob_per_step,
        )
        if gate is not None:
            start &= gate
        if start.any():
            uids = np.nonzero(start)[0]
            disease.on_treatment[uids] = True
            disease.rel_death[uids] = self.rel_death_on_treatment


class ReduceMortalityTx(ChronicCareTx):
    """Backward-compatible wrapper around :class:`ChronicCareTx`."""

    def __init__(
        self,
        disease,
        eligibility_mode="affected",
        rel_death_reduction=0.5,
        *args,
        product=None,
        **kwargs,
    ):
        # Accepted but unused by the historical T2D wrapper.
        del product
        kwargs.pop("prob", None)
        kwargs.pop("eligibility", None)
        disease_key = canonical_dw_key(disease)
        kwargs.setdefault("name", f"{disease_key}_reducemortalitytx")
        service_claim = kwargs.pop("service_claim", None)
        super().__init__(
            disease=disease,
            eligibility=eligibility_mode,
            rel_death_on_treatment=rel_death_reduction,
            service_claim=service_claim,
            *args,
            **kwargs,
        )


def make_ncd_treatment(disease, eligibility_mode="affected", **kwargs) -> ReduceMortalityTx:
    return ReduceMortalityTx(
        disease=disease,
        eligibility_mode=eligibility_mode,
        **kwargs,
    )
