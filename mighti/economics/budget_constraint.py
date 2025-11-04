"""
Budget-Constrained Intervention Framework
Implements time-varying resource allocation and constraint enforcement
compatible with Starsim simulation loops.
"""

import starsim as ss
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from .resource_accounting import ResourcePool
from economics.utils import csv_to_yaml 

__all__ = ["BudgetConstraint"]


class BudgetConstraint(ss.Module):
    """
    Manage and enforce resource constraints (budget, workforce, etc.) during simulation.

    Execution flow:
      1. Initialize resource pools and priority list.
      2. Each timestep, allocate resources to interventions in rank order.
      3. Track DALYs, costs, and HRH usage.
      4. Optionally re-rank interventions based on updated effectiveness.
    """

    def __init__(self, pars=None, enforce=True, priority_file=None, label="budget_constraint"):
        super().__init__(label=label)
        self.pars = pars or {}
        self.enforce = enforce
        self.priority_file = priority_file
        self.resources = None
        self.priority_list = None
        self.history = []  # per-timestep resource usage log


    def init_pre(self, sim):
        """Read parameter file (CSV or YAML), initialize resources and priorities."""
        super().init_pre(sim)

        pars_path = Path(self.pars)

        # --- 1. Auto-convert CSV to YAML internally (in-memory safe mode)
        if pars_path.suffix == ".csv":
            temp_dir = Path("outputs/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            yaml_path = temp_dir / f"{pars_path.stem}.yaml"
            csv_to_yaml(pars_path, yaml_path)

            with open(yaml_path, "r") as f:
                self.pars = yaml.safe_load(f)

            sim.log(f"[BudgetConstraint] Loaded from {pars_path.name} (auto-converted to YAML).")

        # --- 2. Read YAML directly
        elif pars_path.suffix in (".yaml", ".yml"):
            with open(pars_path, "r") as f:
                self.pars = yaml.safe_load(f)

        # --- 3. Direct dict
        elif isinstance(self.pars, dict):
            pass

        else:
            raise ValueError("`pars` must be a path to .csv, .yaml, or a dict of parameters.")

        # --- 4. Initialize resource pool
        self.resources = ResourcePool(
            budget_usd=self.pars.get("budget_usd", float("inf")),
            hrh_minutes=self.pars.get("hrh_minutes", {}),
            rollover=self.pars.get("rollover", True),
        )

        # --- 5. Load or initialize priority list
        if self.priority_file:
            self.priority_list = pd.read_csv(self.priority_file)
        elif "priority_list" in self.pars:
            self.priority_list = pd.DataFrame(self.pars["priority_list"])
        else:
            # Default: all interventions treated equally, alphabetical order
            self.priority_list = pd.DataFrame({
                "intervention": sorted(sim.interventions.keys()),
                "priority": range(len(sim.interventions)),
            })

        self.priority_list.sort_values("priority", inplace=True)

        sim.log(f"[BudgetConstraint] Initialized with total budget ${self.resources.total_budget:,.0f}")


    def register_usage(self, cost, hrh_minutes=None, source=None):
        """Called by interventions to record resource use."""
        hrh_minutes = hrh_minutes or {}
        self.resources.consume(cost=cost, hrh_minutes=hrh_minutes, source=source)


    def has_resources_for(self, cost, hrh_minutes=None):
        """Check feasibility of allocation."""
        return self.resources.has_remaining(cost=cost, hrh_minutes=hrh_minutes)


    def apply(self, sim):
        """
        Execute allocation loop per timestep.
        If enforce=True, halt or scale back when resources depleted.
        """
        timestep_log = dict(time=sim.t, total_cost=0.0, interventions=[])

        for _, row in self.priority_list.iterrows():
            name = row["intervention"]
            if name not in sim.interventions:
                continue

            intv = sim.interventions[name]
            est_cost = getattr(intv, "expected_cost", 0.0)
            est_hrh = getattr(intv, "expected_hrh", {})

            # Stop if budget exhausted
            if self.enforce and not self.has_resources_for(est_cost, est_hrh):
                sim.log(f"[BudgetConstraint] Resource limit reached before {name} at t={sim.t}")
                break

            # Let intervention execute (and internally call register_usage)
            if hasattr(intv, "apply"):
                intv.apply(sim)
            timestep_log["interventions"].append(name)

        timestep_log["total_cost"] = self.resources.used_budget
        self.history.append(timestep_log)


    def finalize_results(self, sim):
        """Summarize cumulative costs and HRH utilization."""
        summary = self.resources.summarize()
        sim.results["budget_constraint"] = summary
        sim.log(f"[BudgetConstraint] Final total cost: ${summary['total_cost']:,.0f}")
        return summary
    