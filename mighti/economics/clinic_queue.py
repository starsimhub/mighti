"""Generic person-level service queue with refillable per-step capacity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import starsim as ss

__all__ = ["QUEUE_RULES", "ClaimRecipe", "ClinicQueue"]

QUEUE_RULES = ("priority", "first_come", "random")


@dataclass(frozen=True)
class ClaimRecipe:
    name: str
    eligibility: object
    slots: float = 1.0
    nurse_min: float = 0.0
    usd: float = 0.0

    def __post_init__(self):
        if not callable(self.eligibility):
            raise TypeError("ClaimRecipe eligibility must be callable")


class ClinicQueue(ss.Intervention):
    """Refill capacity each step and serve caller-defined claims.

    Recipe insertion order defines service order under ``rule="priority"``.
    Eligibility callables receive the simulation and return either a boolean
    mask of length ``n`` or an integer array of UIDs.
    """

    def __init__(
        self,
        recipes,
        slots_per_step=np.inf,
        nurse_min_per_step=np.inf,
        rule="priority",
        rng_seed=1,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("name", "clinic_queue")
        super().__init__(*args, **kwargs)
        rule = (rule or "priority").lower()
        if rule not in QUEUE_RULES:
            raise ValueError(f"Unknown queue rule {rule!r}; choose from {QUEUE_RULES}")
        if not recipes:
            raise ValueError("ClinicQueue requires at least one claim recipe")
        self.slots_per_step = float(slots_per_step)
        self.nurse_min_per_step = float(nurse_min_per_step)
        self.rule = rule
        self.recipes = dict(recipes)
        for name, recipe in self.recipes.items():
            if name != recipe.name:
                raise ValueError(
                    f"Recipe key {name!r} does not match recipe name {recipe.name!r}"
                )
        self.rng_seed = int(rng_seed)
        self._rng = np.random.default_rng(self.rng_seed)
        self.served = {}
        self.wait_days = {}
        self.history = []

    def init_pre(self, sim):
        super().init_pre(sim)
        setattr(sim, "_mighti_clinic_queue", self)
        n = int(sim.people.uid.len_used)
        dt_days = float(getattr(sim.t, "dt_year", 1.0) or 1.0) * 365.25
        self._dt_days = dt_days
        for name in self.recipes:
            self.served[name] = np.zeros(n, dtype=bool)
            self.wait_days[name] = np.zeros(n, dtype=float)

    def _resize(self, n: int) -> None:
        for name in self.recipes:
            if self.served[name].size < n:
                pad = n - self.served[name].size
                self.served[name] = np.pad(self.served[name], (0, pad))
                self.wait_days[name] = np.pad(self.wait_days[name], (0, pad))

    def served_mask(self, claim: str, n: int | None = None) -> np.ndarray:
        """Boolean mask of people served this step for ``claim``."""
        n = int(n if n is not None else self.sim.people.uid.len_used)
        arr = self.served.get(claim)
        if arr is None:
            raise KeyError(f"Unknown queue claim {claim!r}")
        if arr.size < n:
            out = np.zeros(n, dtype=bool)
            out[: arr.size] = arr
            return out
        return arr[:n]

    def _demand_uids(self, sim, claim: str) -> np.ndarray:
        n = int(sim.people.uid.len_used)
        demand = np.asarray(self.recipes[claim].eligibility(sim))
        if demand.dtype == bool:
            if demand.ndim != 1 or demand.size != n:
                raise ValueError(
                    f"Claim {claim!r} eligibility returned boolean shape "
                    f"{demand.shape}; expected ({n},)"
                )
            return np.nonzero(demand)[0]
        uids = np.asarray(demand, dtype=int).reshape(-1)
        if uids.size and ((uids < 0).any() or (uids >= n).any()):
            raise ValueError(f"Claim {claim!r} eligibility returned invalid UIDs")
        return np.unique(uids)

    def _ordered_claims(self, bundles: list[tuple[str, np.ndarray]]) -> list[tuple[str, int]]:
        """Return (claim, uid) in service order."""
        rows: list[tuple[str, int]] = []
        if self.rule == "priority":
            for name, uids in bundles:
                for uid in np.sort(uids):
                    rows.append((name, int(uid)))
            return rows
        for name, uids in bundles:
            for uid in uids:
                rows.append((name, int(uid)))
        if self.rule == "random":
            self._rng.shuffle(rows)
            return rows
        # first_come: stable by uid then claim name
        rows.sort(key=lambda r: (r[1], r[0]))
        return rows

    def step(self):
        sim = self.sim
        n = int(sim.people.uid.len_used)
        self._resize(n)
        for name in self.recipes:
            self.served[name][:] = False

        slots = self.slots_per_step
        nurse = self.nurse_min_per_step
        bundles = [(name, self._demand_uids(sim, name)) for name in self.recipes]
        demand = {name: int(uids.size) for name, uids in bundles}
        served_n = {name: 0 for name in self.recipes}

        for claim, uid in self._ordered_claims(bundles):
            rec = self.recipes[claim]
            if rec.slots > slots or rec.nurse_min > nurse:
                self.wait_days[claim][uid] += self._dt_days
                continue
            slots -= rec.slots
            nurse -= rec.nurse_min
            self.served[claim][uid] = True
            self.wait_days[claim][uid] = 0.0
            served_n[claim] += 1

        # Anyone who demanded and was not served already had wait incremented
        # in the loop except those never reached if we break... we don't break;
        # we skip. Demand not in ordered list cannot happen.

        self.history.append(
            dict(
                ti=int(sim.ti),
                slots_left=float(slots),
                nurse_left=float(nurse),
                demand=demand,
                served=dict(served_n),
                unmet={k: demand[k] - served_n[k] for k in demand},
            )
        )

    def summarize(self) -> dict:
        if not self.history:
            return {}
        unmet = {k: 0 for k in self.recipes}
        demand = {k: 0 for k in self.recipes}
        served = {k: 0 for k in self.recipes}
        for row in self.history:
            for k, v in row["demand"].items():
                demand[k] = demand.get(k, 0) + v
            for k, v in row["served"].items():
                served[k] = served.get(k, 0) + v
            for k, v in row["unmet"].items():
                unmet[k] = unmet.get(k, 0) + v
        wait = {
            k: float(np.mean(self.wait_days[k])) if self.wait_days[k].size else 0.0
            for k in self.recipes
        }
        return dict(
            rule=self.rule,
            slots_per_step=self.slots_per_step,
            demand=demand,
            served=served,
            unmet=unmet,
            mean_wait_days=wait,
        )
