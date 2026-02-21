import pandas as pd
import numpy as np
import starsim as ss
import logging

from mighti.util.rng import get_rng

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

        # logger.info(f"[{self.name}] Initialized module with baseline={self.p_stable:.3f}, inherit={self.inherit_prob:.3f}")

        self.state = None  # will link to People later


    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def init_pre(self, sim):
        """Register the state array on People."""
        super().init_pre(sim)
        # logger.debug(f"[{self.name}] init_pre called at t={sim.t if hasattr(sim, 't') else 'N/A'}")

        if not hasattr(sim.people, self.state_attr):
            arr = ss.BoolArr(self.state_attr)
            sim.people.states.append(arr, overwrite=False)
            setattr(sim.people, self.state_attr, arr)
            arr.link_people(sim.people)
            # logger.debug(f"[{self.name}] State '{self.state_attr}' registered on People")

        self.state = getattr(sim.people, self.state_attr)


    def init_post(self):
        """Assign baseline states after People are fully initialized."""
        super().init_post()
        ppl = self.sim.people
        n = len(ppl)
        if n == 0 or self.state is None:
            # logger.warning(f"[{self.name}] init_post: No people to initialize.")
            return
        rng = get_rng(self.sim, salt=f"{self.__class__.__name__}:init")

        # --- Age-dependent baseline probability ---
        # FIX: Only adjust if explicitly requested (optional future param)
        age = np.array(ppl.age)
        base_prob = np.full(n, self.p_stable)

        # Apply light modulation rather than overwriting
        logistic_mod = 1 / (1 + np.exp(-(age - 15) / 5))
        base_prob = np.clip(base_prob * (0.9 + 0.1 * logistic_mod), 0.05, 0.99)

        self.state[:] = rng.random(n) < base_prob
        prop_stable = np.mean(self.state)
        # logger.info(f"[{self.name}] init_post: {prop_stable:.3f} stable (target {self.p_stable:.3f})")

    def step(self):
        """Yearly inheritance + stochastic transitions, tracking inherited proportion."""
        sim = self.sim
        ppl = sim.people
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")

        # -------------------------------
        # 1. Intergenerational inheritance
        # -------------------------------
        inherited_now = 0
        total_births = 0

        maternal = sim.networks.get("maternalnet", None)
        if maternal is not None and hasattr(maternal, "edges"):
            edges = maternal.edges
            # Starsim edge containers vary by version:
            # - Some provide `edges.start` (timestep when the edge was created)
            # - Others only provide `p1/p2` (mother/baby) and no timing info
            def _to_numpy(x):
                if hasattr(x, "to_numpy"):
                    return x.to_numpy()
                return np.asarray(x)

            try:
                p1 = _to_numpy(edges.p1)
                p2 = _to_numpy(edges.p2)
            except Exception:
                p1 = None
                p2 = None

            if p1 is not None and p2 is not None and len(p1) and len(p2):
                p1 = np.asarray(p1, dtype=float)
                p2 = np.asarray(p2, dtype=float)
                valid = np.isfinite(p1) & np.isfinite(p2)
                if np.any(valid):
                    mothers_all = p1[valid].astype(int, copy=False)
                    babies_all = p2[valid].astype(int, copy=False)

                    # Keep only in-range ids
                    in_range = (mothers_all >= 0) & (mothers_all < len(ppl)) & (babies_all >= 0) & (babies_all < len(ppl))
                    mothers_all = mothers_all[in_range]
                    babies_all = babies_all[in_range]

                    # Identify "newborn edges" for this timestep
                    newborn_mask = None
                    if hasattr(edges, "start"):
                        try:
                            start = _to_numpy(edges.start)
                            start = np.asarray(start, dtype=float)[valid][in_range]
                            newborn_mask = (start == sim.ti)
                        except Exception:
                            newborn_mask = None

                    # Fallback when no edge timing exists: treat babies with age < dt as newborns
                    if newborn_mask is None:
                        dt = float(getattr(sim, "dt", 1.0))
                        ages = _to_numpy(getattr(ppl, "age_years", ppl.age))
                        ages = np.asarray(ages, dtype=float)
                        newborn_mask = ages[babies_all] < (dt + 1e-9)

                    if np.any(newborn_mask):
                        mothers = mothers_all[newborn_mask]
                        babies = babies_all[newborn_mask]
                        n = len(babies)
                        total_births = n
                        inherit_mask = rng.random(n) < self.inherit_prob
                        inherited_now = int(inherit_mask.sum())
                        if inherited_now:
                            self.state[babies[inherit_mask]] = self.state[mothers[inherit_mask]]
                        if n - inherited_now > 0:
                            self.state[babies[~inherit_mask]] = rng.random(n - inherited_now) < self.p_stable

        # -------------------------------
        # 2. Stochastic transitions
        # -------------------------------
        p_loss = getattr(self, "p_loss", 0.01)   # yearly probability of losing stability
        p_gain = getattr(self, "p_gain", 0.10)   # yearly probability of regaining stability

        # Convert StarSim BoolArr → NumPy boolean array
        state_np = np.asarray(self.state, dtype=bool)

        unhoused = ~state_np
        housed   = state_np

        rand_loss = rng.random(len(ppl)) < p_loss
        rand_gain = rng.random(len(ppl)) < p_gain

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

        # if sim.ti % 5 == 0:  # log every 5 years
            # logger.info(
            #     f"[{self.name}] {sim.t.year}: stable={prop_stable:.3f}, "
            #     f"inherited={prop_inherited:.2f}, +{gain_mask.sum()} gained, -{lose_mask.sum()} lost"
            # )
