"""
Module defining hypertension as a chronic disease model.
"""


from dataclasses import dataclass

import numpy as np
import starsim as ss

from mighti.diseases.base_disease import ChronicDisease
from mighti.util.rng import get_rng


__all__ = ["Hypertension", "SBPModel"]


@dataclass(frozen=True)
class SBPModel:
    """
    Caller-supplied contract for continuous systolic blood pressure.

    ``initial_sbp``, ``annual_change``, and ``threshold`` may each be a scalar,
    a population-length array, or a callable with signature ``(sim, uids)``.
    Callables and arrays let downstream projects supply age-, sex-, place-, and
    time-specific values without putting those assumptions in MIGHTI.

    All three fields are required deliberately: MIGHTI does not choose a
    population distribution, natural-history slope, or diagnostic threshold.
    """

    initial_sbp: object
    annual_change: object
    threshold: object


class Hypertension(ChronicDisease):
    """
    Hypertension with an optional continuous-SBP natural-history model.

    With ``sbp_model=None`` (the default), this class retains the existing
    binary ``ChronicDisease`` behavior. Passing an :class:`SBPModel` enables
    continuous physiology:

    * ``sbp`` stores current systolic blood pressure;
    * ``raised_bp`` records whether current SBP meets the caller's threshold;
    * ``affected`` records whether that threshold has ever been crossed; and
    * first crossings populate the existing ``new_cases`` and ``ti_affected``.

    Screening, diagnosis, treatment, adherence, and downstream cardiovascular
    risk are intentionally outside this first physiology-only contract.
    """

    def __init__(self, csv_path, pars=None, sbp_model=None, **kwargs):
        self.disease_name = 'Hypertension'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'Hypertension')
        if sbp_model is not None and not isinstance(sbp_model, SBPModel):
            raise TypeError("sbp_model must be an SBPModel or None")
        self.sbp_model = sbp_model
        self.define_states(
            ss.FloatArr("sbp", default=np.nan, label="Systolic blood pressure"),
            ss.BoolState("raised_bp", default=False, label="Current raised systolic blood pressure"),
        )

    @property
    def uses_continuous_sbp(self):
        """Whether caller-supplied continuous-SBP physiology is enabled."""
        return self.sbp_model is not None

    def _values(self, specification, uids, name):
        """Resolve one caller-supplied scalar, array, or ``(sim, uids)`` callable."""
        uids = np.asarray(uids, dtype=int)
        value = specification(self.sim, uids) if callable(specification) else specification
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            return np.full(uids.size, float(values), dtype=float)
        values = values.reshape(-1)
        if values.size == uids.size:
            return values
        n = int(self.sim.people.uid.len_used)
        if values.size >= n and (uids.size == 0 or int(uids.max()) < values.size):
            return values[uids]
        raise ValueError(
            f"{name} returned {values.size} values for {uids.size} uids "
            f"(population size {n})"
        )

    def _initialize_sbp(self, uids):
        uids = np.asarray(uids, dtype=int)
        if not uids.size:
            return
        initial = self._values(self.sbp_model.initial_sbp, uids, "initial_sbp")
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_sbp must return finite values")
        self.sbp[uids] = initial

    def _threshold(self, uids):
        threshold = self._values(self.sbp_model.threshold, uids, "threshold")
        if not np.all(np.isfinite(threshold)):
            raise ValueError("threshold must return finite values")
        return threshold

    def _sync_pressure_states(self, uids, *, initialize=False):
        """Update current raised BP and irreversible first-crossing history."""
        uids = np.asarray(uids, dtype=int)
        above = np.asarray(self.sbp.raw[uids], dtype=float) >= self._threshold(uids)
        self.raised_bp[uids] = above

        previous = np.asarray(self.affected.raw[uids], dtype=bool)
        new_cases = uids[above & ~previous]
        if initialize:
            self.affected[uids] = above
            self.susceptible[uids] = ~above
            self.at_risk[uids] = ~above
            self.ti_affected[uids[above]] = self.ti
        elif new_cases.size:
            self.set_prognoses(new_cases)
            self.ti_affected[new_cases] = self.ti
        return new_cases

    def init_post(self):
        super().init_post()
        if not self.uses_continuous_sbp:
            return
        uids = np.asarray(self.sim.people.uid, dtype=int)
        self._initialize_sbp(uids)
        self._sync_pressure_states(uids, initialize=True)

    def init_results(self):
        super().init_results()
        if not self.uses_continuous_sbp:
            return
        if "mean_sbp" not in self.results:
            self.define_results(ss.Result("mean_sbp", dtype=float, label="Mean systolic blood pressure"))
        if "raised_bp_prevalence" not in self.results:
            self.define_results(
                ss.Result("raised_bp_prevalence", dtype=float, label="Raised systolic BP prevalence")
            )

    def step(self):
        if not self.uses_continuous_sbp:
            return super().step()

        ti = self.ti
        n = int(self.sim.people.uid.len_used)
        uids = np.arange(n, dtype=int)
        sbp = np.asarray(self.sbp.raw[:n], dtype=float)
        missing = uids[~np.isfinite(sbp)]
        self._initialize_sbp(missing)

        annual_change = self._values(self.sbp_model.annual_change, uids, "annual_change")
        if not np.all(np.isfinite(annual_change)):
            raise ValueError("annual_change must return finite values")
        dt_year = getattr(self.t, "dt_year", None)
        if dt_year is None:
            dt_year = getattr(self.t, "dt", 1.0)
        dt_year = float(getattr(dt_year, "years", dt_year))
        self.sbp[uids] = np.asarray(self.sbp.raw[:n], dtype=float) + annual_change * dt_year
        new_cases = self._sync_pressure_states(uids)

        affected_uids = self.affected.uids
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:step")
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get("p", 0)
        p_death = base_p * rel_death
        if self._competing_enabled():
            self._set_death_pressure(affected_uids, p_death)
            deaths = np.array([], dtype=int)
        else:
            deaths = affected_uids[rng.random(len(affected_uids)) < p_death]
            self.sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected)
        self.results.prevalence[ti] /= max(1, n)
        self.results.mean_sbp[ti] = float(np.mean(np.asarray(self.sbp.raw[:n], dtype=float)))
        self.results.raised_bp_prevalence[ti] = np.count_nonzero(self.raised_bp) / max(1, n)
        return new_cases

