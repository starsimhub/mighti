import pandas as pd
import numpy as np
import starsim as ss
import logging

__all__ = [
    "NeighbourhoodSituation",
    "SocialContext",
    "EducationSituation",
    "EconomicSituation",
    "HealthCareSystem",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------

class BaseSDoH(ss.Module):
    """
    Base module for binary social determinants of health (SDoH) states.

    - Registers a Boolean state array on People (e.g., neighbourhood_situation)
    - Initializes with baseline probability `p_stable`, optionally age-dependent
    - Allows inheritance from mothers via MaternalNet each timestep
    """

    def __init__(
        self,
        name,
        csv_path=None,
        condition_name=None,
        default_p_stable=0.7,
        default_inherit_prob=0.9,
        state_attr=None,
    ):
        super().__init__()
        self.name = name
        self.state_attr = state_attr or f"{name}_situation"

        # Default probabilities
        self.p_stable = default_p_stable
        self.inherit_prob = default_inherit_prob

        # Load probabilities from CSV (if provided)
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

        logger.info(f"[{self.name}] Initialized module with baseline={self.p_stable:.3f}, inherit={self.inherit_prob:.3f}")

        self.state = None  # will link to People later


    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def init_pre(self, sim):
        """Register the state array on People."""
        super().init_pre(sim)
        logger.debug(f"[{self.name}] init_pre called at t={sim.t if hasattr(sim, 't') else 'N/A'}")

        if not hasattr(sim.people, self.state_attr):
            arr = ss.BoolArr(self.state_attr)
            sim.people.states.append(arr, overwrite=False)
            setattr(sim.people, self.state_attr, arr)
            arr.link_people(sim.people)
            logger.debug(f"[{self.name}] State '{self.state_attr}' registered on People")

        self.state = getattr(sim.people, self.state_attr)


    def init_post(self):
        """Assign baseline states after People are fully initialized."""
        super().init_post()
        ppl = self.sim.people
        n = len(ppl)
        if n == 0 or self.state is None:
            logger.warning(f"[{self.name}] init_post: No people to initialize.")
            return

        # --- Age-dependent baseline probability ---
        # FIX: Only adjust if explicitly requested (optional future param)
        age = np.array(ppl.age)
        base_prob = np.full(n, self.p_stable)

        # Apply light modulation rather than overwriting
        logistic_mod = 1 / (1 + np.exp(-(age - 15) / 5))
        base_prob = np.clip(base_prob * (0.9 + 0.1 * logistic_mod), 0.05, 0.99)

        self.state[:] = np.random.rand(n) < base_prob
        prop_stable = np.mean(self.state)
        logger.info(f"[{self.name}] init_post: {prop_stable:.3f} stable (target {self.p_stable:.3f})")
        print(f"[DEBUG] {self.name} init_post — stable={prop_stable:.3f}, expected={self.p_stable:.3f}")

    def step(self):
        """Yearly inheritance + stochastic transitions, tracking inherited proportion."""
        sim = self.sim
        ppl = sim.people

        # -------------------------------
        # 1. Intergenerational inheritance
        # -------------------------------
        inherited_now = 0
        total_births = 0

        maternal = sim.networks.get("maternalnet", None)
        if maternal is not None and hasattr(maternal, "edges"):
            edges = maternal.edges
            new_birth_inds = np.where(edges.start == sim.ti)[0]
            if len(new_birth_inds):
                mothers = edges.p1[new_birth_inds]
                babies = edges.p2[new_birth_inds]
                n = len(babies)
                total_births = n
                inherit_mask = np.random.rand(n) < self.inherit_prob
                inherited_now = inherit_mask.sum()
                if inherited_now:
                    self.state[babies[inherit_mask]] = self.state[mothers[inherit_mask]]
                if n - inherited_now > 0:
                    self.state[babies[~inherit_mask]] = np.random.rand(n - inherited_now) < self.p_stable

        # -------------------------------
        # 2. Stochastic transitions
        # -------------------------------
        p_loss = getattr(self, "p_loss", 0.01)   # yearly probability of losing stability
        p_gain = getattr(self, "p_gain", 0.10)   # yearly probability of regaining stability

        # Convert StarSim BoolArr → NumPy boolean array
        state_np = np.asarray(self.state, dtype=bool)

        unhoused = ~state_np
        housed   = state_np

        rand_loss = np.random.rand(len(ppl)) < p_loss
        rand_gain = np.random.rand(len(ppl)) < p_gain

        lose_mask = housed & rand_loss
        gain_mask = unhoused & rand_gain

        self.state[lose_mask] = False
        self.state[gain_mask] = True

        # -------------------------------
        # 3. Track and log statistics
        # -------------------------------
        if not hasattr(sim, "sdoh_results"):
            sim.sdoh_results = {self.name: {"year": [], "prop_stable": [], "prop_inherited": []}}

        prop_stable = float(np.mean(self.state))
        prop_inherited = (inherited_now / total_births) if total_births > 0 else np.nan

        sim.sdoh_results[self.name]["year"].append(sim.t.year)
        sim.sdoh_results[self.name]["prop_stable"].append(prop_stable)
        sim.sdoh_results[self.name]["prop_inherited"].append(prop_inherited)

        if sim.ti % 5 == 0:  # log every 5 years
            logger.info(
                f"[{self.name}] {sim.t.year}: stable={prop_stable:.3f}, "
                f"inherited={prop_inherited:.2f}, +{gain_mask.sum()} gained, -{lose_mask.sum()} lost"
            )


# ---------------------------------------------------------------------
# Individual SDoH modules
# ---------------------------------------------------------------------

class NeighbourhoodSituation(BaseSDoH):
    """Housing stability / neighbourhood situation (binary)."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="neighbourhood",
            csv_path=csv_path,
            condition_name="NeighbourhoodSituation",
            default_p_stable=0.98,  # 98% housed
            default_inherit_prob=0.9,
            state_attr="neighbourhood_situation",
            **kwargs,
        )
        logger.info(f"[NeighbourhoodSituation] Module initialized with p_stable={self.p_stable:.3f}")


class SocialContext(BaseSDoH):
    """Social support / connectedness."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="social",
            csv_path=csv_path,
            condition_name="SocialContext",
            default_p_stable=0.75,
            default_inherit_prob=0.85,
            state_attr="social_context",
            **kwargs,
        )


class EducationSituation(BaseSDoH):
    """Educational attainment / stability."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="education",
            csv_path=csv_path,
            condition_name="EducationSituation",
            default_p_stable=0.65,
            default_inherit_prob=0.9,
            state_attr="education_situation",
            **kwargs,
        )


class EconomicSituation(BaseSDoH):
    """Economic stability / employment status."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="economic",
            csv_path=csv_path,
            condition_name="EconomicSituation",
            default_p_stable=0.6,
            default_inherit_prob=0.85,
            state_attr="economic_situation",
            **kwargs,
        )


class HealthCareSystem(BaseSDoH):
    """Healthcare access / coverage."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="healthcare",
            csv_path=csv_path,
            condition_name="HealthCareSystem",
            default_p_stable=0.8,
            default_inherit_prob=0.9,
            state_attr="healthcare_system",
            **kwargs,
        )