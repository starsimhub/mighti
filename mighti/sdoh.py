"""
Defines social determinants of health (SDoH) modules for housing,
social context, education, economy, and healthcare systems.
Each module tracks a binary state (stable/unstable) that can be inherited
from the mother at birth via the maternal network.
"""
import pandas as pd
import numpy as np
import starsim as ss

__all__ = [
    "NeighbourhoodSituation",
    "SocialContext",
    "EducationSituation",
    "EconomicSituation",
    "HealthCareSystem",
]


class BaseSDoH(ss.Module):
    """
    Base module for SDoH binary states that can be inherited via MaternalNet.
    Works with Starsim 3.x.
    """

    def __init__(self, name, csv_path=None, condition_name=None,
                 default_p_stable=0.7, default_inherit_prob=0.9,
                 state_attr=None):
        super().__init__()
        self.name = name
        # State array name on People, e.g. "neighbourhood_situation"
        self.state_attr = state_attr or f"{name}_situation"

        self.p_stable = default_p_stable
        self.inherit_prob = default_inherit_prob

        # Load probabilities from CSV if available
        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            condition = (condition_name or name).lower()
            row = df[df["state"].str.strip().str.lower() == condition]
            if not row.empty:
                if "state_prob" in row.columns and pd.notna(row["state_prob"].iloc[0]):
                    self.p_stable = float(row["state_prob"].iloc[0])
                if "inherit_prob" in row.columns and pd.notna(row["inherit_prob"].iloc[0]):
                    self.inherit_prob = float(row["inherit_prob"].iloc[0])

        # Will be bound in init_pre()
        self.state = None

    def init_pre(self, sim):
        """
        Register the state array on People if missing.
        Starsim 3.x: dynamically add arrays via People.states.append(...), link_people(...).
        """
        super().init_pre(sim)

        if not hasattr(sim.people, self.state_attr):
            arr = ss.BoolArr(self.state_attr)                 # create state
            sim.people.states.append(arr, overwrite=False)    # register with People
            setattr(sim.people, self.state_attr, arr)         # attribute access
            arr.link_people(sim.people)                       # link so it resizes with People

        # Keep a direct reference for fast access later
        self.state = getattr(sim.people, self.state_attr)

    def init_post(self):
        """
        Initialize baseline distribution after People are fully initialized.
        """
        super().init_post() 
        n = len(self.sim.people)
        if n == 0 or self.state is None:
            return
        # Randomly assign "stable" according to p_stable
        self.state[:] = np.random.rand(n) < self.p_stable

    def step(self):
        """
        Intergenerational inheritance via MaternalNet for births occurring this timestep.
        """
        maternal = self.sim.networks.get("maternalnet", None)
        if maternal is None or not hasattr(maternal, "edges"):
            return

        edges = maternal.edges
        # births that occur "now"
        if not hasattr(edges, "start"):
            return
        new_birth_inds = np.where(edges.start == self.sim.ti)[0]
        if len(new_birth_inds) == 0:
            return

        mothers = edges.p1[new_birth_inds]
        babies = edges.p2[new_birth_inds]
        n = len(babies)
        if n == 0:
            return

        inherit_mask = np.random.rand(n) < self.inherit_prob
        # Inherit mother's state
        if np.any(inherit_mask):
            self.state[babies[inherit_mask]] = self.state[mothers[inherit_mask]]
        # Otherwise, initialize fresh from baseline
        if np.any(~inherit_mask):
            self.state[babies[~inherit_mask]] = (
                np.random.rand((~inherit_mask).sum()) < self.p_stable
            )

# ---------------------------------------------------------------------
# Individual modules
# ---------------------------------------------------------------------

class NeighbourhoodSituation(BaseSDoH):
    """Housing stability / neighbourhood situation (binary)."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="neighbourhood",
            csv_path=csv_path,
            condition_name="NeighbourhoodSituation",
            default_p_stable=0.7,
            default_inherit_prob=0.9,
            state_attr="neighbourhood_situation",
            **kwargs,
        )


class SocialContext(BaseSDoH):
    """Models social support context."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="social",
            csv_path=csv_path,
            condition_name="SocialContext",
            default_p_stable=0.6,
            default_inherit_prob=0.85,
            **kwargs,
        )

    def get_state_array(self, sim):
        return sim.people.social_context


class EducationSituation(BaseSDoH):
    """Models education attainment stability."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="education",
            csv_path=csv_path,
            condition_name="EducationSituation",
            default_p_stable=0.75,
            default_inherit_prob=0.8,
            **kwargs,
        )

    def get_state_array(self, sim):
        return sim.people.education_situation


class EconomicSituation(BaseSDoH):
    """Models economic/financial stability."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="economic",
            csv_path=csv_path,
            condition_name="EconomicSituation",
            default_p_stable=0.65,
            default_inherit_prob=0.9,
            **kwargs,
        )

    def get_state_array(self, sim):
        return sim.people.economic_situation


class HealthCareSystem(BaseSDoH):
    """Models access to healthcare and system quality."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="healthcare",
            csv_path=csv_path,
            condition_name="HealthCareSystem",
            default_p_stable=0.6,
            default_inherit_prob=0.85,
            **kwargs,
        )

    def get_state_array(self, sim):
        return sim.people.healthcare_system
    