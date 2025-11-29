"""
Defines health conditions and their base logic, including disease-specific behavior and initialization.
"""

import logging
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import lognorm


__all__ = ['RemittingDisease', 'AcuteDisease', 'AcuteSurgicalDisease', 'ChronicDisease',
            'GenericSIS', 'GenericSIR', 'NonAcquiredDisease', 'StaticCondition']


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# ============================================================================
# Severity Management Functions
# ============================================================================

def calculate_severity_weights_option_a(d_gbd, p_mild, p_mod, p_sev, d_mod=None):
    """
    Calculate severity weights using Option A: Symmetric spacing around moderate state.
    
    From the mathematical framework:
    - d_mild = d_mod - Δ
    - d_sev = d_mod + Δ
    - p_mild * d_mild + p_mod * d_mod + p_sev * d_sev = D_GBD
    
    Parameters:
        d_gbd (float): GBD prevalence-weighted average disability weight
        p_mild (float): Proportion of mild cases
        p_mod (float): Proportion of moderate cases
        p_sev (float): Proportion of severe cases
        d_mod (float, optional): Target moderate disability weight. If None, uses d_gbd.
    
    Returns:
        tuple: (d_mild, d_mod, d_sev) disability weights
    """
    if d_mod is None:
        d_mod = d_gbd
    
    # Ensure proportions sum to 1
    total_p = p_mild + p_mod + p_sev
    if abs(total_p - 1.0) > 1e-6:
        logger.warning(f"Severity proportions sum to {total_p}, normalizing to 1.0")
        p_mild /= total_p
        p_mod /= total_p
        p_sev /= total_p
    
    # Calculate Δ from equation (1)
    # p_mild * (d_mod - Δ) + p_mod * d_mod + p_sev * (d_mod + Δ) = D_GBD
    # (p_mild + p_mod + p_sev) * d_mod + (p_sev - p_mild) * Δ = D_GBD
    # d_mod + (p_sev - p_mild) * Δ = D_GBD
    # Δ = (D_GBD - d_mod) / (p_sev - p_mild)
    
    denominator = p_sev - p_mild
    if abs(denominator) < 1e-6:
        # If p_sev ≈ p_mild, use equal spacing
        logger.warning("p_sev ≈ p_mild, using equal spacing around d_mod")
        delta = (d_gbd - d_mod) / (p_mild + p_sev) if (p_mild + p_sev) > 1e-6 else 0.0
    else:
        delta = (d_gbd - d_mod) / denominator
    
    d_mild = max(0.0, d_mod - delta)
    d_sev = min(1.0, d_mod + delta)
    
    # Verify constraint
    weighted_avg = p_mild * d_mild + p_mod * d_mod + p_sev * d_sev
    if abs(weighted_avg - d_gbd) > 1e-3:
        logger.warning(f"Severity weights don't match GBD: {weighted_avg:.4f} vs {d_gbd:.4f}")
    
    return d_mild, d_mod, d_sev


def calculate_severity_weights_option_b(d_gbd, p_mild, p_mod, p_sev, r=0.5, s=2.0):
    """
    Calculate severity weights using Option B: Fixed ratios across severity levels.
    
    From the mathematical framework:
    - d_mild = r * d_mod
    - d_sev = s * d_mod
    - p_mild * d_mild + p_mod * d_mod + p_sev * d_sev = D_GBD
    
    Parameters:
        d_gbd (float): GBD prevalence-weighted average disability weight
        p_mild (float): Proportion of mild cases
        p_mod (float): Proportion of moderate cases
        p_sev (float): Proportion of severe cases
        r (float): Ratio for mild (d_mild = r * d_mod), default 0.5
        s (float): Ratio for severe (d_sev = s * d_mod), default 2.0
    
    Returns:
        tuple: (d_mild, d_mod, d_sev) disability weights
    """
    # Ensure proportions sum to 1
    total_p = p_mild + p_mod + p_sev
    if abs(total_p - 1.0) > 1e-6:
        logger.warning(f"Severity proportions sum to {total_p}, normalizing to 1.0")
        p_mild /= total_p
        p_mod /= total_p
        p_sev /= total_p
    
    # Calculate d_mod from equation (1)
    # d_mod = D_GBD / (p_mild * r + p_mod + p_sev * s)
    denominator = p_mild * r + p_mod + p_sev * s
    if abs(denominator) < 1e-6:
        raise ValueError("Denominator too small in severity weight calculation")
    
    d_mod = d_gbd / denominator
    d_mild = r * d_mod
    d_sev = s * d_mod
    
    # Clip to valid range [0, 1]
    d_mild = max(0.0, min(1.0, d_mild))
    d_mod = max(0.0, min(1.0, d_mod))
    d_sev = max(0.0, min(1.0, d_sev))
    
    # Verify constraint
    weighted_avg = p_mild * d_mild + p_mod * d_mod + p_sev * d_sev
    if abs(weighted_avg - d_gbd) > 1e-3:
        logger.warning(f"Severity weights don't match GBD: {weighted_avg:.4f} vs {d_gbd:.4f}")
    
    return d_mild, d_mod, d_sev


def parse_severity_proportions(proportions_str, n_levels=3):
    """
    Parse severity proportions from a string or list.
    
    Parameters:
        proportions_str: Can be:
            - String like "0.3,0.5,0.2" or "0.3, 0.5, 0.2"
            - List of floats
            - None (returns equal proportions)
        n_levels (int): Number of severity levels (default 3)
    
    Returns:
        np.array: Array of proportions summing to 1.0
    """
    if proportions_str is None:
        # Equal proportions
        return np.ones(n_levels) / n_levels
    
    if isinstance(proportions_str, str):
        # Parse comma-separated string
        parts = [float(x.strip()) for x in proportions_str.split(',')]
    elif isinstance(proportions_str, (list, tuple, np.ndarray)):
        parts = [float(x) for x in proportions_str]
    else:
        raise ValueError(f"Cannot parse severity proportions from {type(proportions_str)}")
    
    if len(parts) != n_levels:
        raise ValueError(f"Expected {n_levels} proportions, got {len(parts)}")
    
    proportions = np.array(parts)
    
    # Normalize to sum to 1
    total = proportions.sum()
    if total < 1e-6:
        logger.warning("Severity proportions sum to near zero, using equal proportions")
        return np.ones(n_levels) / n_levels
    
    return proportions / total


def assign_severity_level(uids, proportions, rng=None):
    """
    Assign severity levels to individuals based on proportions.
    
    Parameters:
        uids (array): Array of individual IDs
        proportions (array): Array of proportions for each severity level (must sum to 1)
        rng (np.random.Generator, optional): Random number generator
    
    Returns:
        np.array: Array of severity levels (0=mild, 1=moderate, 2=severe, etc.)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_levels = len(proportions)
    n_people = len(uids)
    
    # Generate cumulative probabilities
    cumsum = np.cumsum(proportions)
    
    # Assign severity levels
    rand_vals = rng.random(n_people)
    severity_levels = np.digitize(rand_vals, cumsum)
    
    # digitize returns values in [0, n_levels], so clip to [0, n_levels-1]
    severity_levels = np.clip(severity_levels, 0, n_levels - 1)
    
    return severity_levels


def initialize_severity_system(disease, disease_params, pars=None):
    """
    Initialize severity system for a disease.
    
    This function sets up:
    - Number of severity levels (default 3: mild, moderate, severe)
    - Severity proportions for each level
    - Disability weights for each severity level (from severity CSV or calculated from GBD data)
    
    Parameters:
        disease: Disease instance
        disease_params (dict): Parameters from get_disease_parameters()
        pars (dict, optional): Additional parameters that may override defaults
    
    Returns:
        dict: Severity configuration including proportions and weights
    """
    # Check if severity weights are provided directly (from severity CSV)
    if "severity_weights" in disease_params:
        # Direct weights from severity CSV
        severity_weights = disease_params["severity_weights"]
        n_levels = disease_params.get("n_severity_levels", len(severity_weights))
        
        # Default proportions if not specified (equal distribution)
        severity_proportions_str = disease_params.get("severity_proportions", None)
        if pars and "severity_proportions" in pars:
            severity_proportions_str = pars["severity_proportions"]
        
        proportions = parse_severity_proportions(severity_proportions_str, n_levels)
        
        # Store in disease instance
        disease.n_severity_levels = n_levels
        disease.severity_proportions = proportions
        disease.severity_weights = severity_weights
        
        # Store uncertainty bounds if available
        if "severity_weights_uncertainty" in disease_params:
            disease.severity_weights_uncertainty = disease_params["severity_weights_uncertainty"]
        
        # Initialize severity-related parameters for dynamic updates
        # Default: no severity progression (can be overridden)
        if not hasattr(disease, 'severity_progression_rate'):
            disease.severity_progression_rate = pars.get("severity_progression_rate", 0.02) if pars else 0.02
        if not hasattr(disease, 'severity_improvement_rate'):
            disease.severity_improvement_rate = pars.get("severity_improvement_rate", 0.1) if pars else 0.1
        if not hasattr(disease, 'drug_efficacy'):
            disease.drug_efficacy = pars.get("drug_efficacy", 0.9) if pars else 0.9
        if not hasattr(disease, 'severity_mortality_multipliers'):
            # Default: mild=1.0, moderate=1.5, severe=2.5
            if n_levels == 3:
                disease.severity_mortality_multipliers = np.array([1.0, 1.5, 2.5])
            elif n_levels == 2:
                disease.severity_mortality_multipliers = np.array([1.0, 2.0])
            else:
                disease.severity_mortality_multipliers = np.linspace(1.0, 3.0, n_levels)
        
        logger.info(f"Initialized severity system for {disease.disease_name} from severity CSV: "
                    f"{n_levels} levels, proportions={proportions}, weights={severity_weights}")
        
        return {
            "n_levels": n_levels,
            "proportions": proportions,
            "weights": severity_weights
        }
    
    # Otherwise, calculate weights from proportions and GBD data
    # Get number of severity levels
    n_levels = disease_params.get("n_severity_levels", 3)
    if pars and "n_severity_levels" in pars:
        n_levels = pars["n_severity_levels"]
    
    # Get severity proportions
    severity_proportions_str = disease_params.get("severity_proportions", None)
    if pars and "severity_proportions" in pars:
        severity_proportions_str = pars["severity_proportions"]
    
    # Parse proportions
    proportions = parse_severity_proportions(severity_proportions_str, n_levels)
    
    # Calculate disability weights
    d_gbd = disease_params.get("disability_weight_gbd", None)
    if pars and "disability_weight_gbd" in pars:
        d_gbd = pars["disability_weight_gbd"]
    
    severity_method = disease_params.get("severity_method", "option_a")
    if pars and "severity_method" in pars:
        severity_method = pars["severity_method"]
    
    if d_gbd is not None and n_levels == 3:
        # Calculate severity weights using the specified method
        if severity_method == "option_a":
            d_mild, d_mod, d_sev = calculate_severity_weights_option_a(
                d_gbd, proportions[0], proportions[1], proportions[2]
            )
        elif severity_method == "option_b":
            r = pars.get("severity_ratio_mild", 0.5) if pars else 0.5
            s = pars.get("severity_ratio_severe", 2.0) if pars else 2.0
            d_mild, d_mod, d_sev = calculate_severity_weights_option_b(
                d_gbd, proportions[0], proportions[1], proportions[2], r=r, s=s
            )
        else:
            raise ValueError(f"Unknown severity method: {severity_method}")
        
        severity_weights = np.array([d_mild, d_mod, d_sev])
    elif d_gbd is not None and n_levels != 3:
        # For non-3-level systems, distribute weights proportionally
        logger.warning(f"Severity weight calculation for {n_levels} levels not fully implemented, using simple distribution")
        # Simple linear distribution: mild=0.5*d_gbd, moderate=d_gbd, severe=min(1.0, 1.5*d_gbd)
        if n_levels == 1:
            severity_weights = np.array([d_gbd])
        elif n_levels == 2:
            severity_weights = np.array([0.5 * d_gbd, min(1.0, 1.5 * d_gbd)])
        else:
            # For >3 levels, use linear interpolation
            severity_weights = np.linspace(0.3 * d_gbd, min(1.0, 1.7 * d_gbd), n_levels)
    else:
        # No GBD weight provided, use default weights
        logger.warning(f"No GBD disability weight for {disease.disease_name}, using default severity weights")
        if n_levels == 1:
            severity_weights = np.array([0.1])
        elif n_levels == 2:
            severity_weights = np.array([0.05, 0.15])
        elif n_levels == 3:
            severity_weights = np.array([0.03, 0.10, 0.20])
        else:
            severity_weights = np.linspace(0.02, 0.25, n_levels)
    
    # Store in disease instance
    disease.n_severity_levels = n_levels
    disease.severity_proportions = proportions
    disease.severity_weights = severity_weights
    
    # Initialize severity-related parameters for dynamic updates
    if pars:
        disease.severity_progression_rate = pars.get("severity_progression_rate", 0.02)
        disease.severity_improvement_rate = pars.get("severity_improvement_rate", 0.1)
        disease.drug_efficacy = pars.get("drug_efficacy", 0.9)
    else:
        disease.severity_progression_rate = 0.02
        disease.severity_improvement_rate = 0.1
        disease.drug_efficacy = 0.9
    
    # Default severity-based mortality multipliers
    if not hasattr(disease, 'severity_mortality_multipliers'):
        if n_levels == 3:
            disease.severity_mortality_multipliers = np.array([1.0, 1.5, 2.5])
        elif n_levels == 2:
            disease.severity_mortality_multipliers = np.array([1.0, 2.0])
        else:
            disease.severity_mortality_multipliers = np.linspace(1.0, 3.0, n_levels)
    
    logger.info(f"Initialized severity system for {disease.disease_name}: "
                f"{n_levels} levels, proportions={proportions}, weights={severity_weights}")
    
    return {
        "n_levels": n_levels,
        "proportions": proportions,
        "weights": severity_weights
    }


def get_severity_parameters_from_csv(csv_path, disease_name):
    """
    Load severity-specific parameters from a separate severity CSV file.
    
    Expected format:
        condition, mild, moderate, severe, [mild_low, mild_high, ...]
    
    Parameters:
        csv_path (str): Path to the severity CSV file.
        disease_name (str): Name of the disease to look up.
    
    Returns:
        dict: Dictionary with severity weights and proportions, or None if not found.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        if "condition" not in df.columns:
            logger.warning(f"'condition' column missing in severity file: {csv_path}")
            return None
        
        row = df[df["condition"] == disease_name]
        if row.empty:
            logger.debug(f"Disease '{disease_name}' not found in severity file: {csv_path}")
            return None
        
        def get_value_safe(field, default=None):
            if field not in row.columns:
                return default
            val = row[field].values[0]
            if pd.isna(val) or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Get direct disability weights (mild, moderate, severe)
        d_mild = get_value_safe("mild")
        d_mod = get_value_safe("moderate")
        d_sev = get_value_safe("severe")
        
        # If we have all three weights, use them directly
        if d_mild is not None and d_mod is not None and d_sev is not None:
            severity_weights = np.array([d_mild, d_mod, d_sev])
            
            # Get uncertainty bounds if available (for future use)
            d_mild_low = get_value_safe("mild_low")
            d_mild_high = get_value_safe("mild_high")
            d_mod_low = get_value_safe("moderate_low")
            d_mod_high = get_value_safe("moderate_high")
            d_sev_low = get_value_safe("severe_low")
            d_sev_high = get_value_safe("severe_high")
            
            return {
                "severity_weights": severity_weights,
                "n_severity_levels": 3,
                "severity_weights_uncertainty": {
                    "mild": (d_mild_low, d_mild_high),
                    "moderate": (d_mod_low, d_mod_high),
                    "severe": (d_sev_low, d_sev_high),
                } if any(x is not None for x in [d_mild_low, d_mild_high, d_mod_low, d_mod_high, d_sev_low, d_sev_high]) else None,
            }
        
        return None
        
    except FileNotFoundError:
        logger.debug(f"Severity file not found: {csv_path} (this is okay, will use defaults)")
        return None
    except Exception as e:
        logger.warning(f"Error reading severity file {csv_path}: {e}")
        return None


def get_disease_parameters(csv_path, disease_name, severity_csv_path=None):
    """
    Load disease-specific parameters from a CSV file, returning a dictionary
    with required fields and defaults when missing.

    Parameters:
        csv_path (str): Path to the parameter CSV file.
        disease_name (str): Name of the disease to look up.
        severity_csv_path (str, optional): Path to a separate severity CSV file.
            If None, will automatically try to find a severity CSV file based on csv_path.
            For example, if csv_path is "mighti/data/eswatini_parameters.csv",
            it will look for "mighti/data/eswatini_severity.csv".

    Returns:
        dict: Dictionary of parameters for the specified disease.
    """
    import os
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "condition" not in df.columns:
        raise KeyError(f"'condition' column missing in {csv_path}. Available columns: {df.columns.tolist()}")

    row = df[df["condition"] == disease_name]
    if row.empty:
        raise ValueError(f"Disease '{disease_name}' not found in parameter file: {csv_path}")

    def get_value_safe(field, default):
        if field not in row.columns:
            logger.warning(f"Column '{field}' missing for {disease_name}, using default: {default}")
            return default
        val = row[field].values[0]
        if pd.isna(val):
            logger.warning(f"Missing value for '{field}' in {disease_name}, using default: {default}")
            return default
        return val

    # Auto-detect severity CSV file if not provided
    if severity_csv_path is None:
        # Try to construct severity CSV path from parameter CSV path
        # e.g., "mighti/data/eswatini_parameters.csv" -> "mighti/data/eswatini_severity.csv"
        if "_parameters.csv" in csv_path:
            severity_csv_path = csv_path.replace("_parameters.csv", "_severity.csv")
        elif csv_path.endswith("parameters.csv"):
            severity_csv_path = csv_path.replace("parameters.csv", "severity.csv")
        else:
            # Fallback: try in same directory
            dir_path = os.path.dirname(csv_path)
            base_name = os.path.basename(csv_path).replace("_parameters.csv", "").replace("parameters.csv", "")
            if base_name:
                severity_csv_path = os.path.join(dir_path, f"{base_name}_severity.csv")
            else:
                severity_csv_path = os.path.join(dir_path, "severity.csv")

    # Try to load severity parameters from separate CSV file first
    severity_params = None
    if severity_csv_path:
        severity_params = get_severity_parameters_from_csv(severity_csv_path, disease_name)
    
    # If severity CSV not provided or disease not found in it, check main CSV
    if severity_params is None:
        # Severity-related parameters (optional) from main CSV
        n_severity_levels = get_value_safe("n_severity_levels", 3)
        try:
            n_severity_levels = int(n_severity_levels)
        except (ValueError, TypeError):
            n_severity_levels = 3
        
        # Get severity proportions (comma-separated string or individual columns)
        severity_proportions = None
        if "severity_proportions" in row.columns:
            severity_proportions = get_value_safe("severity_proportions", None)
        elif n_severity_levels == 3:
            # Try individual columns for backward compatibility
            p_mild = get_value_safe("p_severity_mild", None)
            p_mod = get_value_safe("p_severity_moderate", None)
            p_sev = get_value_safe("p_severity_severe", None)
            if p_mild is not None and p_mod is not None and p_sev is not None:
                severity_proportions = f"{p_mild},{p_mod},{p_sev}"
        
        # GBD disability weight
        d_gbd = get_value_safe("disability_weight_gbd", None)
        if d_gbd is not None:
            try:
                d_gbd = float(d_gbd)
            except (ValueError, TypeError):
                d_gbd = None
        
        # Severity calculation method
        severity_method = get_value_safe("severity_method", "option_a")  # "option_a" or "option_b"
        
        severity_params = {
            "n_severity_levels": n_severity_levels,
            "severity_proportions": severity_proportions,
            "disability_weight_gbd": d_gbd,
            "severity_method": severity_method,
        }
    
    return {
        "p_death": get_value_safe("p_death", 0.0001),
        "dur_condition": get_value_safe("dur_condition", 10),
        "rel_sus_hiv": get_value_safe("rel_sus", 1.0),
        "remission_rate": get_value_safe("remission_rate", 0.0),
        "max_disease_duration": get_value_safe("max_disease_duration", 30),
        "affected_sex": get_value_safe("affected_sex", "both"),
        "p_acquire": get_value_safe("p_acquire", 0.01),
        # Severity parameters (from separate CSV or main CSV)
        **severity_params,
    }


class RemittingDisease(ss.NCD):
    """ Base class for all remitting diseases."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path    
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)        
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        # Calculate the mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        # Define parameters using extracted values
        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),  # Log-normal distribution for duration
            p_death=ss.bernoulli(disease_params["p_death"]),  
            remission_rate=disease_params["remission_rate"],  
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],  
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire=disease_params["p_acquire"],
            init_prev=None
        )
        
        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate) 

        self.update_pars(pars, **kwargs)
        
        # Initialize severity system
        initialize_severity_system(self, disease_params, pars)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),   
            ss.BoolState('affected'),
            ss.BoolState('on_treatment'),
            ss.BoolState('reversed'), 
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_dead'), 
            ss.FloatArr('rel_sus', default=1.0),  
            ss.FloatArr('rel_death', default=1.0),
            ss.IntArr('severity_level', default=0),  # 0=mild, 1=moderate, 2=severe, etc.
            reset=True,
        )

    def init_post(self):

        super().init_post()

        # (1) initialize baseline risk if relevant
        initial_risk = self.pars['initial_risk'].filter()
        self.at_risk[initial_risk] = True
        self.ti_affected[initial_risk] = self.ti + self.pars['dur_risk'].rvs(initial_risk, round=True)

        # (2) initialize prevalence
        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            probs = self.pars.init_prev.rvs(self.sim.people.uid)          # ← fixed
            affected = np.random.rand(len(self.sim.people)) < probs       # ← fixed

            if hasattr(self, "affected"):
                self.affected[:] = affected

            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True
        # Assign severity levels to new cases
        if len(uids) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(uids, self.severity_proportions)
            self.severity_level[uids] = severity_levels

    def init_results(self):
        super().init_results()
        existing_results = set(self.results.keys())

        if 'new_cases' not in existing_results:
            self.define_results(ss.Result('new_cases', dtype=int, label='New Cases'))
        if 'new_deaths' not in existing_results:
            self.define_results(ss.Result('new_deaths', dtype=int, label='Deaths'))
        if 'prevalence' not in existing_results:
            self.define_results(ss.Result('prevalence', dtype=float, label='Prevalence'))
        if 'remission_prevalence' not in existing_results:
            self.define_results(ss.Result('remission_prevalence', dtype=float, label='Remission Prevalence'))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)

    def step_state(self):
        if hasattr(self, "p_remission"):
            going_into_remission = self.p_remission.filter(self.affected.uids) 
            self.affected[going_into_remission] = False
            self.reversed[going_into_remission] = True
            self.ti_reversed[going_into_remission] = self.ti

            recovered = (self.reversed & (self.ti_reversed <= self.ti)).uids
            self.reversed[recovered] = False
            self.susceptible[recovered] = True
        
        # Update severity dynamically based on treatment effectiveness
        if hasattr(self, 'severity_level') and hasattr(self, 'affected'):
            affected_uids = self.affected.uids
            if len(affected_uids) > 0:
                try:
                    update_severity_dynamic(self, self.sim, affected_uids)
                except Exception as e:
                    logger.debug(f"Severity update failed for {self.disease_name}: {e}")  

    def step(self):
        ti = self.ti

        susceptible = (~self.affected).uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(self.sim.people, 'hiv'):
                hiv_pos = self.sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
            
            # Apply severity-based acquisition multiplier (from upstream diseases)
            severity_acq_mult = get_severity_acquisition_multiplier(self, susceptible)
            p_acq *= severity_acq_mult
        except Exception:
            pass

        draws = np.random.rand(len(susceptible))
        new_cases = susceptible[draws < p_acq]

        self.affected[new_cases] = True
        self.ti_affected[new_cases] = ti
        
        # Assign severity levels to new cases
        if len(new_cases) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(new_cases, self.severity_proportions)
            self.severity_level[new_cases] = severity_levels

        # Dynamic death logic — allows rel_death to be changed over time
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        
        # Apply severity-based mortality multiplier
        if hasattr(self, 'severity_level'):
            severity_mortality_mult = get_severity_mortality_multiplier(self, affected_uids)
            rel_death = rel_death * severity_mortality_mult

        try:
            base_p = self.pars.p_death.pars['p']  # extract base death prob
        except Exception:
            raise ValueError(f"Cannot extract base death probability from {self.pars.p_death}")

        adjusted_p_death = base_p * rel_death
        draws = np.random.rand(len(affected_uids))
        deaths = affected_uids[draws < adjusted_p_death]
        self.ti_dead[deaths] = ti  

        self.sim.people.request_death(deaths)
        self.results.new_deaths[ti] = len(deaths)

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        self.results.remission_prevalence[self.ti] = np.count_nonzero(self.reversed) / len(self.sim.people)
        return new_cases

    @property
    def duration(self):
        """Duration of active condition in years, with NaN-safety."""
        if not hasattr(self, 'ti_affected') or not hasattr(self, 'affected'):
            raise AttributeError("This disease does not support duration")

        n = len(self.sim.people)
        dur = np.zeros(n)
        ti_now = self.ti

        # Defensive copy and clean any nan or invalid times
        ti_aff = np.asarray(self.ti_affected, dtype=float)
        ti_aff[~np.isfinite(ti_aff)] = 0.0

        # active indices that exist within current population size
        active = self.affected.uids[self.affected.uids < n]
        if len(active):
            dur[active] = np.maximum(0, ti_now - ti_aff[active])

        # Replace any remaining NaN with 0
        dur[~np.isfinite(dur)] = 0.0
        return dur


class AcuteDisease(ss.NCD):
    """Base class for all acute diseases."""

    def __init__(self, csv_path=None, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        # Calculate mean in log-space (mu)
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire=disease_params["p_acquire"],
            init_prev=None,
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.update_pars(pars, **kwargs)
        
        # Initialize severity system
        initialize_severity_system(self, disease_params, pars)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),
            ss.BoolState('affected'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            ss.IntArr('severity_level', default=0),  # 0=mild, 1=moderate, 2=severe, etc.
            reset=True,
        )

    def init_post(self):
        
        super().init_post()
        sim = self.sim

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            # Sample prevalence probabilities
            probs = self.pars.init_prev.rvs(sim.people.uid)
            affected = np.random.rand(len(sim.people)) < probs

            # Assign disease state
            if hasattr(self, "affected"):
                self.affected[:] = affected

            # Optionally set prognoses for affected agents
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True
        self.at_risk[uids] = False
        # Assign severity levels to new cases
        if len(uids) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(uids, self.severity_proportions)
            self.severity_level[uids] = severity_levels

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases', int, 'New Cases'),
            ('new_deaths', int, 'Deaths'),
            ('prevalence', float, 'Prevalence')
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        susceptible = self.at_risk.uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(self.sim.people, 'hiv'):
                hiv_pos = self.sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]

        self.sim.people.request_death(deaths)
        self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return new_cases


    @property
    def duration(self):
        """
        Duration (in years) since onset of disease, 0 if not affected.
        This allows YLD calculations in MicrocostingAnalyzer.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)

        # Handle different onset attributes
        if hasattr(self, 'affected') and hasattr(self, 'ti_affected'):
            affected_uids = self.affected.uids
            if len(affected_uids):
                dur[affected_uids] = self.sim.t.years - self.ti_affected[affected_uids]
        elif hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            infected_uids = self.infected.uids
            if len(infected_uids):
                dur[infected_uids] = self.sim.t.years - self.ti_infected[infected_uids]

        # Clip negatives (e.g. from pre-sim infections)
        dur = np.clip(dur, 0, None)
        return dur
    


class AcuteSurgicalDisease(ss.NCD):
    """
    Acute disease with a possible surgical intervention event.

    Represents conditions like appendicitis, congenital heart anomalies, or digestive congenital anomalies
    that are acute in course but can be surgically treated to improve survival.

    Parameters loaded from CSV include:
        - dur_condition: mean duration of untreated disease (yrs)
        - p_death: baseline probability of death per timestep
        - p_acquire: per-timestep acquisition probability
        - p_surgery: probability of receiving surgery
        - rel_mortality_treated: relative mortality for treated individuals
        - rel_mortality_untreated: relative mortality for untreated individuals
        - cost_surgery (optional): for MicrocostingAnalyzer integration
    """

    def __init__(self, csv_path=None, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params.get("rel_sus_hiv", 1.0),
            affected_sex=disease_params.get("affected_sex", "both"),
            p_acquire_multiplier=1.0,
            p_acquire=disease_params["p_acquire"],
            p_surgery=disease_params.get("p_surgery", 0.3),
            rel_mortality_treated=disease_params.get("rel_mortality_treated", 0.5),
            rel_mortality_untreated=disease_params.get("rel_mortality_untreated", 2.0),
            cost_surgery=disease_params.get("cost_surgery", 0.0),
            init_prev=None,
        )

        self.p_acquire = ss.bernoulli(
            p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids)
        )
        self.update_pars(pars, **kwargs)
        
        # Initialize severity system
        initialize_severity_system(self, disease_params, pars)

        # Define states
        self.define_states(
            ss.BoolState("susceptible", default=True),
            ss.BoolState("at_risk", default=True),
            ss.BoolState("affected"),
            ss.BoolState("on_treatment"),   # here: has received surgery
            ss.BoolState("surgery_done", default=False),
            ss.FloatArr("ti_affected"),
            ss.FloatArr("ti_dead"),
            ss.FloatArr("ti_surgery"),
            ss.FloatArr("rel_sus", default=1.0),
            ss.FloatArr("rel_death", default=1.0),
            ss.IntArr("severity_level", default=0),  # 0=mild, 1=moderate, 2=severe, etc.
            reset=True,
        )

    def init_post(self):
        super().init_post()
        sim = self.sim

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            probs = self.pars.init_prev.rvs(sim.people.uid)
            affected = np.random.rand(len(sim.people)) < probs
            if hasattr(self, "affected"):
                self.affected[:] = affected
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])
        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True
        self.at_risk[uids] = False
        self.rel_death[uids] = self.pars.rel_mortality_untreated
        # Assign severity levels to new cases
        if len(uids) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(uids, self.severity_proportions)
            self.severity_level[uids] = severity_levels

    def init_results(self):
        super().init_results()
        existing_results = set(self.results.keys())

        if "new_cases" not in existing_results:
            self.define_results(ss.Result("new_cases", dtype=int, label="New Cases"))
        if "new_deaths" not in existing_results:
            self.define_results(ss.Result("new_deaths", dtype=int, label="Deaths"))
        if "new_surgeries" not in existing_results:
            self.define_results(ss.Result("new_surgeries", dtype=int, label="Surgeries"))
        if "prevalence" not in existing_results:
            self.define_results(ss.Result("prevalence", dtype=float, label="Prevalence"))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        sim = self.sim

        # --- Acquisition ---
        susceptible = self.at_risk.uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(sim.people, "hiv"):
                hiv_pos = sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti
        self.rel_death[new_cases] = self.pars.rel_mortality_untreated
        
        # Assign severity levels to new cases
        if len(new_cases) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(new_cases, self.severity_proportions)
            self.severity_level[new_cases] = severity_levels

        # --- Surgery events ---
        affected_uids = self.affected.uids
        can_surgery = affected_uids[~self.surgery_done[affected_uids]]
        surgeries = can_surgery[np.random.rand(len(can_surgery)) < self.pars.p_surgery]
        if len(surgeries):
            self.on_treatment[surgeries] = True
            self.surgery_done[surgeries] = True
            self.ti_surgery[surgeries] = ti
            self.rel_death[surgeries] = self.pars.rel_mortality_treated

        # --- Deaths ---
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        base_p = self.pars.p_death.pars.get("p", 0)
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]
        if len(deaths):
            sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        # --- Results ---
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_surgeries[ti] = len(surgeries)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)

        return new_cases

    @property
    def duration(self):
        """Duration (in years) since onset of disease."""
        n = len(self.sim.people)
        dur = np.zeros(n)
        if hasattr(self, "affected") and hasattr(self, "ti_affected"):
            affected_uids = self.affected.uids
            if len(affected_uids):
                dur[affected_uids] = self.sim.t.years - self.ti_affected[affected_uids]
        dur = np.clip(dur, 0, None)
        return dur
    

class ChronicDisease(ss.NCD):
    """Base class for chronic diseases."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire=disease_params["p_acquire"],
            init_prev=None,
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.update_pars(pars, **kwargs)
        
        # Initialize severity system
        initialize_severity_system(self, disease_params, pars)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),
            ss.BoolState('affected'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_affected'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            ss.IntArr('severity_level', default=0),  # 0=mild, 1=moderate, 2=severe, etc.
            reset=True,
        )

    def init_post(self):
 
        super().init_post()
        sim = self.sim 

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            # Sample prevalence probabilities
            probs = self.pars.init_prev.rvs(sim.people.uid)
            affected = np.random.rand(len(sim.people)) < probs

            # Assign disease state
            if hasattr(self, "affected"):
                self.affected[:] = affected

            # Optionally set prognoses for affected agents
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(affected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.affected[uids] = True
        self.at_risk[uids] = False
        # Assign severity levels to new cases
        if len(uids) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(uids, self.severity_proportions)
            self.severity_level[uids] = severity_levels

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases', int, 'New Cases'),
            ('new_deaths', int, 'Deaths'),
            ('prevalence', float, 'Prevalence')
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        susceptible = self.at_risk.uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        if self.pars.affected_sex == "female":
            p_acq[self.sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[self.sim.people.female[susceptible]] = 0

        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(self.sim.people, 'hiv'):
                hiv_pos = self.sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
            
            # Apply severity-based acquisition multiplier (from upstream diseases)
            severity_acq_mult = get_severity_acquisition_multiplier(self, susceptible)
            p_acq *= severity_acq_mult
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        self.affected[new_cases] = True
        self.at_risk[new_cases] = False
        self.ti_affected[new_cases] = ti
        
        # Assign severity levels to new cases
        if len(new_cases) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(new_cases, self.severity_proportions)
            self.severity_level[new_cases] = severity_levels

        # Update severity dynamically based on treatment effectiveness
        if hasattr(self, 'severity_level') and hasattr(self, 'affected'):
            affected_uids_for_severity = self.affected.uids
            if len(affected_uids_for_severity) > 0:
                try:
                    update_severity_dynamic(self, self.sim, affected_uids_for_severity)
                except Exception as e:
                    logger.debug(f"Severity update failed for {self.disease_name}: {e}")

        # Deaths
        affected_uids = self.affected.uids
        rel_death = self.rel_death[affected_uids]
        
        # Apply severity-based mortality multiplier
        if hasattr(self, 'severity_level'):
            severity_mortality_mult = get_severity_mortality_multiplier(self, affected_uids)
            rel_death = rel_death * severity_mortality_mult
        
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]

        self.sim.people.request_death(deaths)
        self.ti_dead[deaths] = ti

        # Results
        self.results.new_cases[ti] = len(new_cases)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[self.ti] = np.count_nonzero(self.affected) / len(self.sim.people)
        return new_cases

    @property
    def duration(self):
        """
        Duration (in years) since onset of disease, 0 if not affected.
        This allows YLD calculations in MicrocostingAnalyzer.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)

        # Handle different onset attributes
        if hasattr(self, 'affected') and hasattr(self, 'ti_affected'):
            affected_uids = self.affected.uids
            if len(affected_uids):
                dur[affected_uids] = self.sim.t.years - self.ti_affected[affected_uids]
        elif hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            infected_uids = self.infected.uids
            if len(infected_uids):
                dur[infected_uids] = self.sim.t.years - self.ti_infected[infected_uids]

        # Clip negatives (e.g. from pre-sim infections)
        dur = np.clip(dur, 0, None)
        return dur
    

class GenericSIS(ss.SIS):
    """Base class for communicable diseases (SIS model)."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            remission_rate=disease_params["remission_rate"],
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire=disease_params["p_acquire"],
            init_prev=pars.get("init_prev", ss.bernoulli(0)) if pars else ss.bernoulli(0),
            # Severity tracking and multipliers
            track_severity_from=pars.get("track_severity_from", None) if pars else None,  # Name of disease to track severity from (e.g., 'hiv')
            severity_acquisition_per_level=pars.get("severity_acquisition_per_level", 0.0) if pars else 0.0,  # Multiplier per severity level (e.g., 0.5 = 50% increase per level)
            severity_mortality_per_level=pars.get("severity_mortality_per_level", 0.0) if pars else 0.0,  # Multiplier per severity level (e.g., 0.4 = 40% increase per level)
        )

        self.p_acquire = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate)
        self.update_pars(pars, **kwargs)
        
        # Initialize severity system
        initialize_severity_system(self, disease_params, pars)

        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),
            ss.BoolState('infected'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('ti_reversed'),
            ss.FloatArr('ti_recovered'),  # Alias for ti_reversed (required by base ss.SIS.step_state)
            ss.FloatArr('ti_dead'),
            ss.FloatArr('immunity', default=0.0),  # Required by base ss.SIS.update_immunity()
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            ss.IntArr('severity_level', default=0),  # 0=mild, 1=moderate, 2=severe, etc.
            reset=True,
        )

    def init_post(self):
        super().init_post()

        sim = self.sim  # Starsim assigns this automatically in init_pre(sim)

        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            # Sample prevalence probabilities
            probs = self.pars.init_prev.rvs(sim.people.uid)
            infected = np.random.rand(len(sim.people)) < probs

            # Assign disease state
            if hasattr(self, "infected"):
                self.infected[:] = infected 

            # Optionally set prognoses for infected agents
            if hasattr(self, "set_prognoses"):
                self.set_prognoses(np.where(infected)[0])

        return

    def set_prognoses(self, uids):
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.at_risk[uids] = False
        # Assign severity levels to new cases
        if len(uids) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(uids, self.severity_proportions)
            self.severity_level[uids] = severity_levels

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases', int, 'New Cases'),
            ('new_deaths', int, 'Deaths'),
            ('prevalence', float, 'Prevalence'),
            ('recovered', int, 'New Recoveries')
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        sim = self.sim
        
        # --- Acquire infection (S → I) ---
        susceptible = self.at_risk.uids & self.susceptible.uids
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        # Sex filtering
        if self.pars.affected_sex == "female":
            p_acq[sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[sim.people.female[susceptible]] = 0

        # Modifiers
        try:
            p_acq *= self.rel_sus[susceptible]
            
            # HIV interaction: apply base rel_sus_hiv
            if hasattr(sim.people, 'hiv'):
                hiv_pos = sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
            
            # Apply severity-based acquisition multiplier (from upstream diseases)
            # If severity_acquisition_per_level is set, use custom formula
            if self.pars.severity_acquisition_per_level > 0:
                severity_acq_mult = get_severity_acquisition_multiplier(
                    self, susceptible, 
                    per_level_multiplier=self.pars.severity_acquisition_per_level
                )
                p_acq *= severity_acq_mult
            else:
                # Use default severity acquisition multiplier
                severity_acq_mult = get_severity_acquisition_multiplier(self, susceptible)
                p_acq *= severity_acq_mult
        except Exception as e:
            logger.debug(f"Acquisition multiplier failed for {self.disease_name}: {e}")

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        if len(new_cases):
            self.infected[new_cases] = True
            self.susceptible[new_cases] = False
            self.at_risk[new_cases] = False
            self.ti_infected[new_cases] = ti
            
            # Assign severity levels to new cases
            # If track_severity_from is set, use that disease's severity for people with that disease
            if self.pars.track_severity_from and hasattr(sim.diseases, self.pars.track_severity_from):
                tracked_disease = getattr(sim.diseases, self.pars.track_severity_from)
                if hasattr(tracked_disease, 'severity_level'):
                    # Check which new cases have the tracked disease (e.g., HIV+)
                    if hasattr(sim.people, self.pars.track_severity_from):
                        has_tracked = getattr(sim.people, self.pars.track_severity_from)[new_cases]
                    elif hasattr(tracked_disease, 'infected'):
                        has_tracked = tracked_disease.infected[new_cases]
                    else:
                        has_tracked = np.zeros(len(new_cases), dtype=bool)
                    
                    # For people with tracked disease: copy their severity
                    # For people without: use default assignment
                    if hasattr(self, 'severity_proportions'):
                        severity_levels = assign_severity_level(new_cases, self.severity_proportions)
                        self.severity_level[new_cases] = severity_levels
                    
                    if np.any(has_tracked):
                        tracked_severity = tracked_disease.severity_level[new_cases[has_tracked]]
                        # Clip to valid range for this disease
                        n_levels = self.n_severity_levels if hasattr(self, 'n_severity_levels') else 3
                        tracked_severity = np.clip(tracked_severity, 0, n_levels - 1)
                        self.severity_level[new_cases[has_tracked]] = tracked_severity
                elif hasattr(self, 'severity_proportions'):
                    severity_levels = assign_severity_level(new_cases, self.severity_proportions)
                    self.severity_level[new_cases] = severity_levels
            elif hasattr(self, 'severity_proportions'):
                severity_levels = assign_severity_level(new_cases, self.severity_proportions)
                self.severity_level[new_cases] = severity_levels

        # --- Recoveries (I → S, SIS model allows reinfection) ---
        infected_uids = self.infected.uids
        new_rec = np.array([], dtype=int)  # Initialize empty array
        if len(infected_uids) > 0:
            # p_remission.filter() returns UIDs directly, not indices
            new_rec = self.p_remission.filter(infected_uids)
            if len(new_rec):
                self.infected[new_rec] = False
                self.susceptible[new_rec] = True
                self.at_risk[new_rec] = True
                # Set both ti_reversed and ti_recovered (base ss.SIS expects ti_recovered)
                if hasattr(self, 'ti_reversed'):
                    self.ti_reversed[new_rec] = ti
                if hasattr(self, 'ti_recovered'):
                    self.ti_recovered[new_rec] = ti

        # Update severity dynamically
        if hasattr(self, 'severity_level') and len(infected_uids) > 0:
            try:
                # If track_severity_from is set, update severity to match tracked disease for people with that disease
                if self.pars.track_severity_from and hasattr(sim.diseases, self.pars.track_severity_from):
                    tracked_disease = getattr(sim.diseases, self.pars.track_severity_from)
                    if hasattr(tracked_disease, 'severity_level'):
                        # Check which infected individuals have the tracked disease (e.g., HIV+)
                        if hasattr(sim.people, self.pars.track_severity_from):
                            has_tracked = getattr(sim.people, self.pars.track_severity_from)[infected_uids]
                        elif hasattr(tracked_disease, 'infected'):
                            has_tracked = tracked_disease.infected[infected_uids]
                        else:
                            has_tracked = np.zeros(len(infected_uids), dtype=bool)
                        
                        # For people with tracked disease: copy their severity
                        if np.any(has_tracked):
                            tracked_severity = tracked_disease.severity_level[infected_uids[has_tracked]]
                            # Clip to valid range
                            n_levels = self.n_severity_levels if hasattr(self, 'n_severity_levels') else 3
                            tracked_severity = np.clip(tracked_severity, 0, n_levels - 1)
                            self.severity_level[infected_uids[has_tracked]] = tracked_severity
                        
                        # For people without tracked disease: use dynamic update
                        if np.any(~has_tracked):
                            update_severity_dynamic(self, sim, infected_uids[~has_tracked])
                    else:
                        update_severity_dynamic(self, sim, infected_uids)
                else:
                    update_severity_dynamic(self, sim, infected_uids)
            except Exception as e:
                logger.debug(f"Severity update failed for {self.disease_name}: {e}")

        # --- Deaths among infected ---
        rel_death = self.rel_death[infected_uids] if len(infected_uids) else np.array([])
        
        # Apply severity-based mortality multiplier
        if hasattr(self, 'severity_level') and len(infected_uids) > 0:
            # If severity_mortality_per_level is set, use custom formula
            if self.pars.severity_mortality_per_level > 0:
                severity_levels = self.severity_level[infected_uids]
                # Formula: multiplier = 1.0 + per_level * (severity - 1)
                # For severity=1 (mild): multiplier = 1.0
                # For severity=2 (moderate): multiplier = 1.0 + per_level
                # For severity=3 (severe): multiplier = 1.0 + 2*per_level
                severity_mortality_mult = 1.0 + self.pars.severity_mortality_per_level * (severity_levels - 1)
            else:
                # Use default severity mortality multiplier
                severity_mortality_mult = get_severity_mortality_multiplier(self, infected_uids)
            
            if len(rel_death) > 0:
                rel_death = rel_death * severity_mortality_mult
            else:
                rel_death = severity_mortality_mult
        
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = infected_uids[np.random.rand(len(infected_uids)) < base_p * (rel_death if len(rel_death) else 1.0)]
        if len(deaths):
            sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        # --- Results ---
        self.results.new_cases[ti] = len(new_cases)
        self.results.recovered[ti] = len(new_rec)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.infected) / len(sim.people)

        return new_cases
    
    @property
    def duration(self):
        """
        Duration (in years) since onset of disease, 0 if not affected.
        This allows YLD calculations in MicrocostingAnalyzer.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)

        # Handle different onset attributes
        if hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            affected_uids = self.infected.uids
            if len(affected_uids):
                # Filter to only include UIDs within current population bounds
                # (some people may have died, so their UIDs are out of bounds)
                valid_mask = affected_uids < n
                affected_uids = affected_uids[valid_mask]
                
                if len(affected_uids):
                    # Convert timesteps to years: ti_infected is stored in timesteps
                    # Get current year from timeline
                    if hasattr(self.sim.t, 'year'):
                        current_year = self.sim.t.year
                    elif hasattr(self.sim.t, 'yearvec'):
                        current_year = self.sim.t.yearvec[self.ti]
                    else:
                        current_year = self.ti  # Fallback to timestep
                    
                    # Convert ti_infected from timesteps to years
                    if hasattr(self.sim.t, 'yearvec'):
                        ti_infected_years = self.sim.t.yearvec[self.ti_infected[affected_uids].astype(int)]
                    else:
                        # Fallback: assume dt=1 (yearly timesteps)
                        ti_infected_years = self.ti_infected[affected_uids]
                    
                    dur[affected_uids] = current_year - ti_infected_years
        elif hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            infected_uids = self.infected.uids
            if len(infected_uids):
                # Filter to only include UIDs within current population bounds
                valid_mask = infected_uids < n
                infected_uids = infected_uids[valid_mask]
                
                if len(infected_uids):
                    # Convert timesteps to years
                    if hasattr(self.sim.t, 'year'):
                        current_year = self.sim.t.year
                    elif hasattr(self.sim.t, 'yearvec'):
                        current_year = self.sim.t.yearvec[self.ti]
                    else:
                        current_year = self.ti
                    
                    if hasattr(self.sim.t, 'yearvec'):
                        ti_infected_years = self.sim.t.yearvec[self.ti_infected[infected_uids].astype(int)]
                    else:
                        ti_infected_years = self.ti_infected[infected_uids]
                    
                    dur[infected_uids] = current_year - ti_infected_years

        # Clip negatives (e.g. from pre-sim infections)
        dur = np.clip(dur, 0, None)
        return dur


class GenericSIR(ss.SIR):
    """Base class for communicable diseases (SIR model)."""

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)
        
        sigma = 0.5
        mu = np.log(disease_params["dur_condition"]) - (sigma**2) / 2

        self.define_pars(
            dur_condition=lognorm(s=sigma, scale=np.exp(mu)),
            p_death=ss.bernoulli(disease_params["p_death"]),
            remission_rate=disease_params["remission_rate"],   # per-timestep recovery prob
            max_disease_duration=disease_params["max_disease_duration"],
            rel_sus_hiv=disease_params["rel_sus_hiv"],
            affected_sex=disease_params["affected_sex"],
            p_acquire_multiplier=1.0,
            p_acquire=disease_params["p_acquire"],             # force of infection term
            init_prev=pars.get("init_prev", ss.bernoulli(0)) if pars else ss.bernoulli(0),
        )

        # Stochastic processes
        self.p_acquire   = ss.bernoulli(p=lambda self, sim, uids: calculate_p_acquire_generic(self, sim, uids))
        self.p_remission = ss.bernoulli(p=lambda self, sim, uids: self.pars.remission_rate)

        self.update_pars(pars, **kwargs)
        
        # Initialize severity system
        initialize_severity_system(self, disease_params, pars)

        # States
        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('at_risk', default=True),     # convenience mask for who can acquire
            ss.BoolState('infected'),
            ss.BoolState('recovered'),
            ss.BoolState('on_treatment'),
            ss.FloatArr('ti_infected'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('rel_sus',   default=1.0),
            ss.FloatArr('rel_death', default=1.0),
            ss.IntArr('severity_level', default=0),  # 0=mild, 1=moderate, 2=severe, etc.
            reset=True,
        )

    def init_post(self):
        super().init_post()
        sim = self.sim

        # Initialize infection prevalence if provided
        if hasattr(self.pars, "init_prev") and callable(getattr(self.pars.init_prev, "rvs", None)):
            probs = self.pars.init_prev.rvs(sim.people.uid)
            init_inf = np.random.rand(len(sim.people)) < probs

            self.infected[init_inf]   = True
            self.susceptible[init_inf] = False
            self.at_risk[init_inf]     = False
            self.ti_infected[init_inf] = self.ti

    def set_prognoses(self, uids):
        # Enter I from S
        self.susceptible[uids] = False
        self.infected[uids]    = True
        self.at_risk[uids]     = False
        # Assign severity levels to new cases
        if len(uids) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(uids, self.severity_proportions)
            self.severity_level[uids] = severity_levels

    def init_results(self):
        super().init_results()
        for name, dtype, label in [
            ('new_cases',   int,   'New Cases'),
            ('new_deaths',  int,   'Deaths'),
            ('prevalence',  float, 'Prevalence (Infected)'),
            ('recovered',   int,   'New Recoveries'),
        ]:
            if name not in self.results:
                self.define_results(ss.Result(name, dtype=dtype, label=label))

    def update_results(self):
        super().update_results()
        self.results.prevalence[self.ti] = np.count_nonzero(self.infected) / len(self.sim.people)

    def step(self):
        ti = self.ti
        sim = self.sim

        # --- Acquire infection (S → I) ---
        susceptible = self.at_risk.uids & self.susceptible.uids  # ensure truly in S
        p_acq = np.full(len(susceptible), self.pars.p_acquire_multiplier * self.pars.p_acquire)

        # Sex filtering
        if self.pars.affected_sex == "female":
            p_acq[sim.people.male[susceptible]] = 0
        elif self.pars.affected_sex == "male":
            p_acq[sim.people.female[susceptible]] = 0

        # Modifiers
        try:
            p_acq *= self.rel_sus[susceptible]
            if hasattr(sim.people, 'hiv'):
                hiv_pos = sim.people.hiv[susceptible]
                p_acq[hiv_pos] *= self.pars.rel_sus_hiv
            
            # Apply severity-based acquisition multiplier (from upstream diseases)
            severity_acq_mult = get_severity_acquisition_multiplier(self, susceptible)
            p_acq *= severity_acq_mult
        except Exception:
            pass

        new_cases = susceptible[np.random.rand(len(susceptible)) < p_acq]
        if len(new_cases):
            self.infected[new_cases]    = True
            self.susceptible[new_cases] = False
            self.at_risk[new_cases]     = False
            self.ti_infected[new_cases] = ti
            
            # Assign severity levels to new cases
            if hasattr(self, 'severity_proportions'):
                severity_levels = assign_severity_level(new_cases, self.severity_proportions)
                self.severity_level[new_cases] = severity_levels

        # --- Recoveries (I → R, no reinfection in classic SIR) ---
        infected_uids = self.infected.uids
        # p_remission.filter() returns UIDs directly, not indices
        new_rec = self.p_remission.filter(infected_uids)
        if len(new_rec):
            self.infected[new_rec]  = False
            self.recovered[new_rec] = True
            self.ti_recovered[new_rec] = ti

        # Update severity dynamically based on treatment effectiveness
        if hasattr(self, 'severity_level') and len(infected_uids) > 0:
            try:
                update_severity_dynamic(self, sim, infected_uids)
            except Exception as e:
                logger.debug(f"Severity update failed for {self.disease_name}: {e}")

        # --- Deaths among infected (optional relative risk) ---
        rel_death = self.rel_death[infected_uids] if len(infected_uids) else np.array([])
        
        # Apply severity-based mortality multiplier
        if hasattr(self, 'severity_level') and len(infected_uids) > 0:
            severity_mortality_mult = get_severity_mortality_multiplier(self, infected_uids)
            if len(rel_death) > 0:
                rel_death = rel_death * severity_mortality_mult
            else:
                rel_death = severity_mortality_mult
        
        base_p = self.pars.p_death.pars.get('p', 0)
        deaths = infected_uids[np.random.rand(len(infected_uids)) < base_p * (rel_death if len(rel_death) else 1.0)]
        if len(deaths):
            sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        # --- Results ---
        self.results.new_cases[ti]  = len(new_cases)
        self.results.recovered[ti]  = len(new_rec)
        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.infected) / len(sim.people)

        return new_cases

    @property
    def duration(self):
        """
        Duration (years) since infection onset for currently infected;
        recovered or susceptible return 0. Useful for YLD calculations.
        """
        n = len(self.sim.people)
        dur = np.zeros(n)
        if hasattr(self, 'infected') and hasattr(self, 'ti_infected'):
            iu = self.infected.uids
            if len(iu):
                # Filter to only include UIDs within current population bounds
                valid_mask = iu < n
                iu = iu[valid_mask]
                
                if len(iu):
                    # Convert timesteps to years: ti_infected is stored in timesteps
                    if hasattr(self.sim.t, 'year'):
                        current_year = self.sim.t.year
                    elif hasattr(self.sim.t, 'yearvec'):
                        current_year = self.sim.t.yearvec[self.ti]
                    else:
                        current_year = self.ti
                    
                    # Convert ti_infected from timesteps to years
                    if hasattr(self.sim.t, 'yearvec'):
                        ti_infected_years = self.sim.t.yearvec[self.ti_infected[iu].astype(int)]
                    else:
                        ti_infected_years = self.ti_infected[iu]
                    
                    dur[iu] = current_year - ti_infected_years
        dur = np.clip(dur, 0, None)
        return dur
    
    
class NonAcquiredDisease(ss.Module):
    """
    Base class for congenital or neonatal (non-acquired) diseases.

    Used for:
        - Neonatal conditions (encephalopathy, preterm birth, sepsis)
        - Congenital anomalies (heart, limb, digestive)
        - Static genetic disorders (Down Syndrome, Chromosomal Abnormalities)

    Features:
        - No acquisition or remission processes
        - No 'at_risk' or 'susceptible' states
        - Static prevalence initialized at birth
        - Optional neonatal restriction (<28 days)
        - Mortality via p_death
    """
    depends_on = ["Deaths", "DeathsExtended"]

    def __init__(self, csv_path, pars=None, is_neonatal=False, **kwargs):
        super().__init__()
        self.csv_path = csv_path
        self.is_neonatal = is_neonatal
        self.disease_name = getattr(self, "disease_name", self.__class__.__name__)

        # Load parameters
        disease_params = get_disease_parameters(csv_path=self.csv_path, disease_name=self.disease_name)

        # Define parameters (no acquisition or remission)
        self.define_pars(
            p_death=ss.bernoulli(disease_params.get("p_death", 0.0)),
            dur_condition=disease_params.get("dur_condition", 1.0),
            max_disease_duration=disease_params.get("max_disease_duration", 1.0),
            rel_sus_hiv=disease_params.get("rel_sus_hiv", 1.0),
            affected_sex=disease_params.get("affected_sex", "both"),
            init_prev=pars.get("init_prev", ss.bernoulli(0.01)) if pars else ss.bernoulli(0.01),
        )
        self.update_pars(pars, **kwargs)
        
        # Initialize severity system
        initialize_severity_system(self, disease_params, pars)

        # Define minimal states
        self.define_states(
            ss.BoolState("affected", default=False, label="Affected"),
            ss.FloatArr("rel_death", default=1.0, label="Relative mortality multiplier"),
            ss.FloatArr("rel_sus", default=1.0, label="Relative susceptibility"),
            ss.FloatArr("ti_affected", label="Time of becoming affected"),
            ss.FloatArr("ti_dead", label="Time of death"),
            ss.IntArr("severity_level", default=0, label="Severity level"),  # 0=mild, 1=moderate, 2=severe, etc.
            reset=True,
        )

    # ---------------------------------------------------------------------
    # Initialization lifecycle
    # ---------------------------------------------------------------------
    def init_pre(self, sim):
        super().init_pre(sim)
        return

    def init_post(self):
        """Initialize congenital/neonatal prevalence at birth."""
        super().init_post()
        sim = self.sim
        n = len(sim.people)

        # Draw initial affected status
        if hasattr(self.pars.init_prev, "rvs"):
            affected = self.pars.init_prev.rvs(sim.people.uid)
        elif callable(self.pars.init_prev):
            affected = np.array(self.pars.init_prev(), dtype=bool)
        else:
            p = float(self.pars.init_prev)
            affected = np.random.rand(n) < p

        self.affected[:] = affected
        self.ti_affected[affected] = self.ti
        
        # Assign severity levels to affected individuals
        affected_uids = np.where(affected)[0]
        if len(affected_uids) > 0 and hasattr(self, 'severity_proportions'):
            severity_levels = assign_severity_level(affected_uids, self.severity_proportions)
            self.severity_level[affected_uids] = severity_levels

        n_affected = affected.sum()
        logger.info(f"[INIT] {self.disease_name}: {n_affected}/{n} ({n_affected/n:.3%}) affected at birth")

    def init_results(self):
        super().init_results()
        existing = set(self.results.keys())
        if "prevalence" not in existing:
            self.define_results(ss.Result("prevalence", dtype=float, scale=False, label="Prevalence"))
        if "new_deaths" not in existing:
            self.define_results(ss.Result("new_deaths", dtype=int, scale=True, label="Deaths"))
        if "n_affected" not in existing:
            self.define_results(ss.Result("n_affected", dtype=int, scale=False, label="Affected"))

    # ---------------------------------------------------------------------
    # Step logic
    # ---------------------------------------------------------------------
    def step_state(self):
        """No within-step state transitions for congenital diseases."""
        return

    def step(self):
        """Apply mortality among affected individuals."""
        # Skip the very first timestep so neonatal deaths are not applied before survivorship baseline
        if self.ti == 0:
            return

        ti = self.ti
        sim = self.sim
        affected_uids = self.affected.uids
        if not len(affected_uids):
            return

        # Restrict to neonates if needed
        if self.is_neonatal:
            ages = getattr(sim.people, "age_years", sim.people.age)
            affected_uids = affected_uids[ages[affected_uids] < (28 / 365)]
            if not len(affected_uids):
                return

        # Update severity dynamically based on treatment effectiveness
        if hasattr(self, 'severity_level') and len(affected_uids) > 0:
            try:
                update_severity_dynamic(self, sim, affected_uids)
            except Exception as e:
                logger.debug(f"Severity update failed for {self.disease_name}: {e}")

        base_p = self.pars.p_death.pars.get("p", 0)
        rel_death = self.rel_death[affected_uids]
        
        # Apply severity-based mortality multiplier
        if hasattr(self, 'severity_level'):
            severity_mortality_mult = get_severity_mortality_multiplier(self, affected_uids)
            rel_death = rel_death * severity_mortality_mult
        
        deaths = affected_uids[np.random.rand(len(affected_uids)) < base_p * rel_death]

        if len(deaths):
            sim.people.request_death(deaths)
            logger.debug(f"[STEP] {self.disease_name}: {len(deaths)} deaths at timestep {ti}")

        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        self.results.n_affected[ti] = np.count_nonzero(self.affected)

    def step_die(self, uids):
        """Record death times for affected individuals."""
        if len(uids):
            affected_dead = uids[self.affected[uids]]
            if len(affected_dead):
                self.ti_dead[affected_dead] = self.sim.people.ti_dead[affected_dead]
        return

    # ---------------------------------------------------------------------
    # Results tracking
    # ---------------------------------------------------------------------
    def update_results(self):
        super().update_results()
        ti = self.ti
        sim = self.sim
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        self.results.n_affected[ti] = np.count_nonzero(self.affected)

    # ---------------------------------------------------------------------
    # Finalization lifecycle
    # ---------------------------------------------------------------------
    def finalize(self):
        super().finalize()
        ppl = self.sim.people
        dead = ppl.dead.uids
        affected_dead = dead[self.affected[dead]]
        if len(affected_dead):
            self.ti_dead[affected_dead] = ppl.ti_dead[affected_dead]
        logger.debug(f"[FINAL] {self.disease_name}: {len(affected_dead)} total deaths recorded.")

    def finalize_results(self):
        super().finalize_results()

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def duration(self):
        """Duration (years) of condition presence since birth."""
        n = len(self.sim.people)
        dur = np.zeros(n)
        affected_uids = self.affected.uids
        if len(affected_uids):
            # Filter to only include UIDs within current population bounds
            valid_mask = affected_uids < n
            affected_uids = affected_uids[valid_mask]
            
            if len(affected_uids):
                # Get current time in years
                if hasattr(self.sim.t, 'year'):
                    current_time = self.sim.t.year
                elif hasattr(self.sim.t, 'yearvec'):
                    current_time = self.sim.t.yearvec[self.ti]
                else:
                    current_time = self.ti
                
                # Convert ti_affected from timesteps to years
                if hasattr(self.sim.t, 'yearvec'):
                    ti_affected_years = self.sim.t.yearvec[self.ti_affected[affected_uids].astype(int)]
                else:
                    ti_affected_years = self.ti_affected[affected_uids]
                
                dur[affected_uids] = current_time - ti_affected_years
        return dur
    

class StaticCondition(NonAcquiredDisease):
    """
    Lifelong static (non-progressive) conditions like Down Syndrome or chromosomal abnormalities.
    """

    def __init__(self, csv_path, pars=None, **kwargs):
        super().__init__(csv_path, pars, is_neonatal=False, **kwargs)
        self.define_pars(dur_condition=np.inf, max_disease_duration=np.inf)

    def step(self):
        """Lifelong condition — mortality only, no remission."""
        ti = self.ti
        sim = self.sim
        affected = self.affected.uids
        if not len(affected):
            return np.array([])

        # Update severity dynamically based on treatment effectiveness
        if hasattr(self, 'severity_level') and len(affected) > 0:
            try:
                update_severity_dynamic(self, sim, affected)
            except Exception as e:
                logger.debug(f"Severity update failed for {self.disease_name}: {e}")

        rel_death = self.rel_death[affected]
        
        # Apply severity-based mortality multiplier
        if hasattr(self, 'severity_level'):
            severity_mortality_mult = get_severity_mortality_multiplier(self, affected)
            rel_death = rel_death * severity_mortality_mult
        
        base_p = self.pars.p_death.pars.get("p", 0)
        deaths = affected[np.random.rand(len(affected)) < base_p * rel_death]
        if len(deaths):
            sim.people.request_death(deaths)
            self.ti_dead[deaths] = ti

        self.results.new_deaths[ti] = len(deaths)
        self.results.prevalence[ti] = np.count_nonzero(self.affected) / len(sim.people)
        return deaths    

def calculate_treatment_effectiveness(disease, sim, uids):
    """
    Calculate treatment effectiveness for individuals based on:
    Effectiveness = Adherence × DrugEfficacy × BiologicalPotency
    
    Parameters:
        disease: Disease instance
        sim: Simulation object
        uids: Array of individual IDs
    
    Returns:
        np.array: Treatment effectiveness values (0-1) for each individual
    """
    n = len(uids)
    if n == 0:
        return np.array([])
    
    # Initialize effectiveness components
    adherence = np.ones(n)  # Default: perfect adherence
    drug_efficacy = 1.0  # Default: 100% efficacy
    biological_potency = 1.0  # Default: normal potency
    
    # Get adherence from CASM connector if available
    if hasattr(sim, 'connectors'):
        # Try to find CASM adherence connector
        casm_connector = None
        if isinstance(sim.connectors, dict):
            for key, conn in sim.connectors.items():
                if 'casm' in key.lower() and 'adherence' in key.lower():
                    casm_connector = conn
                    break
        else:
            for conn in sim.connectors:
                if hasattr(conn, 'label') and 'casm' in conn.label.lower() and 'adherence' in conn.label.lower():
                    casm_connector = conn
                    break
        
        if casm_connector is not None:
            # Get adherence values from connector
            # The connector may store adherence per intervention
            # For now, use a simplified approach: check if intervention has rel_effect
            try:
                # Try to get adherence from interventions
                for intv in sim.interventions:
                    if hasattr(intv, 'rel_effect') and hasattr(intv, 'casm_sensitivity'):
                        # Get adherence for this intervention
                        if hasattr(intv, 'adherence') or hasattr(intv, 'adherence_scale'):
                            # Use intervention-specific adherence
                            adherence_val = getattr(intv, 'adherence_scale', 1.0)
                            if hasattr(intv, 'adherence'):
                                # If adherence is per-person, extract for these uids
                                intv_adherence = getattr(intv, 'adherence', None)
                                if intv_adherence is not None:
                                    if hasattr(intv_adherence, '__getitem__'):
                                        adherence = np.array([intv_adherence[uid] if uid < len(intv_adherence) else 1.0 for uid in uids])
                                    else:
                                        adherence = np.full(n, float(intv_adherence))
                                else:
                                    adherence = np.full(n, adherence_val)
                            else:
                                adherence = np.full(n, adherence_val)
                            break
            except Exception as e:
                logger.debug(f"Could not get adherence from connector: {e}")
    
    # Get drug efficacy from disease parameters or intervention
    if hasattr(disease, 'pars') and hasattr(disease.pars, 'drug_efficacy'):
        drug_efficacy = float(disease.pars.drug_efficacy)
    elif hasattr(disease, 'drug_efficacy'):
        drug_efficacy = float(disease.drug_efficacy)
    else:
        # Default: check if on treatment and use default efficacy
        if hasattr(disease, 'on_treatment'):
            # For people on treatment, use default efficacy (can be overridden)
            drug_efficacy = 0.9  # Default 90% efficacy
    
    # Get biological potency (can be individual-specific or constant)
    if hasattr(disease, 'pars') and hasattr(disease.pars, 'biological_potency'):
        if hasattr(disease.pars.biological_potency, '__getitem__'):
            biological_potency = np.array([disease.pars.biological_potency[uid] if uid < len(disease.pars.biological_potency) else 1.0 for uid in uids])
        else:
            biological_potency = np.full(n, float(disease.pars.biological_potency))
    elif hasattr(disease, 'biological_potency'):
        if hasattr(disease.biological_potency, '__getitem__'):
            biological_potency = np.array([disease.biological_potency[uid] if uid < len(disease.biological_potency) else 1.0 for uid in uids])
        else:
            biological_potency = np.full(n, float(disease.biological_potency))
    else:
        biological_potency = np.ones(n)  # Default: normal potency
    
    # Ensure biological_potency is array
    if not isinstance(biological_potency, np.ndarray):
        biological_potency = np.full(n, float(biological_potency))
    
    # Calculate effectiveness: Effectiveness = Adherence × DrugEfficacy × BiologicalPotency
    effectiveness = adherence * drug_efficacy * biological_potency
    
    # Clip to [0, 1]
    effectiveness = np.clip(effectiveness, 0.0, 1.0)
    
    return effectiveness


def update_severity_dynamic(disease, sim, uids, treatment_effectiveness=None):
    """
    Update severity levels dynamically based on treatment effectiveness and disease progression.
    
    Severity can improve (decrease) with effective treatment or worsen (increase) without treatment.
    
    Parameters:
        disease: Disease instance
        sim: Simulation object
        uids: Array of individual IDs to update
        treatment_effectiveness: Optional pre-calculated effectiveness array. If None, will calculate.
    
    Returns:
        np.array: Updated severity levels
    """
    if len(uids) == 0:
        return np.array([])
    
    if not hasattr(disease, 'severity_level'):
        logger.warning(f"Disease {disease.disease_name} does not have severity_level state")
        return np.array([])
    
    n_levels = disease.n_severity_levels if hasattr(disease, 'n_severity_levels') else 3
    current_severity = disease.severity_level[uids].copy()
    
    # Get treatment effectiveness
    if treatment_effectiveness is None:
        treatment_effectiveness = calculate_treatment_effectiveness(disease, sim, uids)
    
    # Check who is on treatment
    on_treatment = np.zeros(len(uids), dtype=bool)
    if hasattr(disease, 'on_treatment'):
        on_treatment = disease.on_treatment[uids]
    
    # Severity update rules:
    # - With effective treatment: severity tends to decrease (improve)
    # - Without treatment or ineffective treatment: severity may increase (worsen) or stay same
    # - Natural progression: severity may increase over time
    
    # Get disease duration for progression
    if hasattr(disease, 'duration'):
        durations = disease.duration[uids]
    else:
        durations = np.zeros(len(uids))
    
    # Calculate severity change probability
    # High effectiveness → more likely to improve
    # Low/no effectiveness → more likely to worsen or stay same
    # Longer duration → more likely to progress (worsen) if untreated
    
    new_severity = current_severity.copy()
    
    for i, uid in enumerate(uids):
        current_level = current_severity[i]
        effectiveness = treatment_effectiveness[i] if i < len(treatment_effectiveness) else 0.0
        is_treated = on_treatment[i] if i < len(on_treatment) else False
        duration = durations[i] if i < len(durations) else 0.0
        
        # Probability of improvement (decrease severity)
        if is_treated and effectiveness > 0.5:
            # Effective treatment: high chance of improvement
            p_improve = effectiveness * 0.1  # 10% chance per timestep at 100% effectiveness
            if np.random.rand() < p_improve and current_level > 0:
                new_severity[i] = current_level - 1
        
        # Probability of worsening (increase severity)
        if not is_treated or effectiveness < 0.3:
            # No treatment or ineffective treatment: risk of worsening
            # Base progression rate increases with duration
            p_worsen = 0.02 * (1 + duration / 10)  # 2% base, increases with duration
            if np.random.rand() < p_worsen and current_level < (n_levels - 1):
                new_severity[i] = current_level + 1
    
    # Update severity levels
    disease.severity_level[uids] = new_severity
    
    return new_severity


def get_severity_mortality_multiplier(disease, uids=None):
    """
    Get mortality risk multipliers based on severity levels.
    Higher severity → higher mortality risk.
    
    Parameters:
        disease: Disease instance
        uids: Array of individual IDs. If None, uses all affected individuals.
    
    Returns:
        np.array: Mortality multipliers (typically > 1.0 for higher severity)
    """
    if not hasattr(disease, 'severity_level'):
        return np.ones(len(uids) if uids is not None else 0)
    
    if uids is None:
        if hasattr(disease, 'affected'):
            uids = disease.affected.uids
        elif hasattr(disease, 'infected'):
            uids = disease.infected.uids
        else:
            return np.array([])
    
    if len(uids) == 0:
        return np.array([])
    
    severity_levels = disease.severity_level[uids]
    n_levels = disease.n_severity_levels if hasattr(disease, 'n_severity_levels') else 3
    
    # Default multipliers: mild=1.0, moderate=1.5, severe=2.5
    # Can be overridden by disease-specific parameters
    if hasattr(disease, 'severity_mortality_multipliers'):
        multipliers = disease.severity_mortality_multipliers
    else:
        if n_levels == 1:
            multipliers = np.array([1.0])
        elif n_levels == 2:
            multipliers = np.array([1.0, 2.0])
        elif n_levels == 3:
            multipliers = np.array([1.0, 1.5, 2.5])
        else:
            # Linear interpolation
            multipliers = np.linspace(1.0, 3.0, n_levels)
    
    # Map severity levels to multipliers
    severity_levels = np.clip(severity_levels, 0, len(multipliers) - 1)
    return multipliers[severity_levels]


def get_severity_acquisition_multiplier(disease, uids=None, per_level_multiplier=0.0):
    """
    Get acquisition probability multipliers based on severity of upstream diseases.
    Higher severity in one disease → higher risk of acquiring other diseases.
    
    Parameters:
        disease: Disease instance (the disease being acquired)
        uids: Array of individual IDs. If None, uses all susceptible individuals.
        per_level_multiplier: If > 0, use formula: 1.0 + per_level * (severity - 1)
            For example, 0.5 means 50% increase per severity level above 1.
            If 0, uses default multipliers (mild=1.0, moderate=1.2, severe=1.5)
    
    Returns:
        np.array: Acquisition multipliers (typically > 1.0 for higher upstream severity)
    """
    if uids is None:
        if hasattr(disease, 'susceptible'):
            uids = disease.susceptible.uids
        elif hasattr(disease, 'at_risk'):
            uids = disease.at_risk.uids
        else:
            return np.ones(0)
    
    if len(uids) == 0:
        return np.ones(0)
    
    sim = disease.sim
    multipliers = np.ones(len(uids))
    
    # Check for upstream diseases that affect acquisition
    # For example, severe HIV increases risk of TB, pneumonia, etc.
    if hasattr(sim, 'diseases'):
        # Check HIV severity if HIV exists
        if hasattr(sim.people, 'hiv') and hasattr(sim.diseases, 'hiv'):
            hiv = sim.diseases.hiv
            if hasattr(hiv, 'severity_level'):
                hiv_severity = hiv.severity_level[uids]
                hiv_infected = sim.people.hiv[uids]
                
                # Higher HIV severity → higher acquisition risk
                # Use disease-specific rel_sus_hiv parameter (already applied in step())
                # Here we only apply severity-based multiplier
                
                if per_level_multiplier > 0:
                    # Custom formula: multiplier = 1.0 + per_level * (severity - 1)
                    # For severity=1 (mild): multiplier = 1.0
                    # For severity=2 (moderate): multiplier = 1.0 + per_level
                    # For severity=3 (severe): multiplier = 1.0 + 2*per_level
                    severity_mult = np.ones(len(uids))
                    severity_mult[hiv_infected] = 1.0 + per_level_multiplier * (hiv_severity[hiv_infected] - 1)
                else:
                    # Default severity-based multiplier: mild=1.0, moderate=1.2, severe=1.5
                    severity_mult = np.ones(len(uids))
                    severity_mult[hiv_infected & (hiv_severity == 1)] = 1.2  # Moderate
                    severity_mult[hiv_infected & (hiv_severity == 2)] = 1.5  # Severe
                
                multipliers *= severity_mult
    
    return multipliers


def get_disability_weight_by_severity(disease, uids=None):
    """
    Get disability weights for individuals based on their severity levels.
    
    Parameters:
        disease: Disease instance with severity_level state
        uids (array, optional): Array of individual IDs. If None, returns weights for all affected individuals.
    
    Returns:
        np.array: Array of disability weights corresponding to each individual's severity level
    """
    if not hasattr(disease, 'severity_weights'):
        # Fallback: return a default weight if severity system not initialized
        logger.warning(f"Severity system not initialized for {disease.disease_name}, using default weight")
        if uids is None:
            # Try to get affected individuals
            if hasattr(disease, 'affected'):
                uids = disease.affected.uids
            elif hasattr(disease, 'infected'):
                uids = disease.infected.uids
            else:
                return np.array([])
        return np.full(len(uids), 0.1)  # Default weight
    
    if uids is None:
        # Get all affected/infected individuals
        if hasattr(disease, 'affected'):
            uids = disease.affected.uids
        elif hasattr(disease, 'infected'):
            uids = disease.infected.uids
        else:
            return np.array([])
    
    if len(uids) == 0:
        return np.array([])
    
    # Get severity levels for these individuals
    severity_levels = disease.severity_level[uids]
    
    # Map severity levels to weights (clip to valid range)
    severity_levels = np.clip(severity_levels, 0, len(disease.severity_weights) - 1)
    
    # Return corresponding disability weights
    return disease.severity_weights[severity_levels]


def calculate_p_acquire_generic(disease, sim, uids):
    """Calculate acquisition probability for a disease with optional sex filtering and HIV interaction."""
    p_base = np.full(len(uids), disease.pars.p_acquire_multiplier * disease.pars.p_acquire)
    
    if disease.pars.affected_sex == "female":
        try:
            p_base[sim.people.male[uids]] = 0
        except Exception:
            pass
    elif disease.pars.affected_sex == "male":
        try:
            p_base[sim.people.female[uids]] = 0
        except Exception:
            pass

    try:
        if hasattr(sim.people, 'hiv'):
            hiv_positive = sim.people.hiv[uids]
            p_base[hiv_positive] *= disease.pars.rel_sus_hiv
    except Exception:
        pass

    try:
        return p_base * disease.rel_sus[uids]
    except Exception:
        return p_base       
    