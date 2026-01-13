"""
Tracks budget and workforce resource pools for BudgetConstraint.
"""

import copy

__all__ = ["ResourcePool"]


class ResourcePool:
    """
    Class for tracking and updating financial and workforce resource pools.

    Attributes
    ----------
    total_budget : float
        Total available budget at initialization.
    remaining_budget : float
        Current remaining budget.
    hrh_minutes : dict
        Total available human-resource minutes per cadre (e.g., doctor, nurse).
    remaining_hrh : dict
        Remaining minutes for each cadre.
    rollover : bool
        Whether unspent budget and HRH minutes roll over to the next timestep.
    used_budget : float
        Cumulative budget spent to date.
    log : list[dict]
        Log of all resource usage events.
    """

    def __init__(self, budget_usd, hrh_minutes=None, rollover=True):
        self.total_budget = float(budget_usd or 0)
        self.remaining_budget = float(budget_usd or 0)
        self.hrh_minutes = copy.deepcopy(hrh_minutes or {})
        self.remaining_hrh = copy.deepcopy(hrh_minutes or {})
        self.rollover = bool(rollover)
        self.used_budget = 0.0
        self.log = []

    # -------------------------------------------------------------------------
    def consume(self, cost, hrh_minutes=None, source=None):
        """
        Deduct cost and HRH use for a given intervention or activity.

        Parameters
        ----------
        cost : float
            Amount of budget to deduct.
        hrh_minutes : dict
            Dict of cadre: minutes used.
        source : str
            Name of the intervention or module using resources.

        Returns
        -------
        bool
            True if allocation succeeded, False if insufficient resources.
        """
        hrh_minutes = hrh_minutes or {}
        if not self.has_remaining(cost, hrh_minutes):
            return False

        # Deduct from pools
        self.remaining_budget = max(0.0, self.remaining_budget - cost)
        self.used_budget += cost
        for cadre, mins in hrh_minutes.items():
            if cadre in self.remaining_hrh:
                self.remaining_hrh[cadre] = max(0.0, self.remaining_hrh[cadre] - mins)

        # Log the event
        self.log.append(dict(source=source, cost=cost, hrh=copy.deepcopy(hrh_minutes)))
        return True

    # -------------------------------------------------------------------------
    def has_remaining(self, cost, hrh_minutes=None):
        """Return True if enough budget and HRH remain to allocate."""
        if cost > self.remaining_budget:
            return False
        hrh_minutes = hrh_minutes or {}
        for cadre, mins in hrh_minutes.items():
            if cadre in self.remaining_hrh and mins > self.remaining_hrh[cadre]:
                return False
        return True

    # -------------------------------------------------------------------------
    def reset_rollover(self):
        """
        Apply rollover at the end of a timestep.
        If rollover=False, reset remaining resources to initial totals.
        """
        if not self.rollover:
            self.remaining_budget = self.total_budget
            self.remaining_hrh = copy.deepcopy(self.hrh_minutes)

    # -------------------------------------------------------------------------
    def summarize(self):
        """Return dictionary summary of usage and remaining resources."""
        total_hrh = sum(self.hrh_minutes.values()) if self.hrh_minutes else 0
        used_hrh = {k: self.hrh_minutes[k] - self.remaining_hrh.get(k, 0)
                    for k in self.hrh_minutes.keys()}
        return dict(
            total_cost=self.used_budget,
            total_budget=self.total_budget,
            remaining_budget=self.remaining_budget,
            hrh_total=total_hrh,
            hrh_used=used_hrh,
            hrh_remaining=self.remaining_hrh,
            log=self.log,
        )
    