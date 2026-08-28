"""
Default disability weights for MIGHTI CEA / MicrocostingAnalyzer.

Phase-1 convention: one average GBD health-state weight per modeled condition
(severity-specific DWs are parked). Values are provisional proxies pending
external sign-off; override per run via ``disability_weights={...}``.

Keys are lowercase canonical names used by ``resolve_disease_module``.
Aliases (e.g. ``t2d``, ``Type2Diabetes``) are accepted by helpers below.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_GBD_DISABILITY_WEIGHTS",
    "DW_ALIASES",
    "DW_METADATA",
    "CORE_CEA_CONDITIONS",
    "MULTIMORBIDITY_RULES",
    "DEFAULT_MULTIMORBIDITY_RULE",
    "SENSITIVITY_MULTIMORBIDITY_RULE",
    "HIV_YLD_MODES",
    "DEFAULT_HIV_YLD_MODE",
    "HIV_STAGE_DISABILITY_WEIGHTS",
    "HIV_STAGE_METADATA",
    "HIV_CD4_AIDS",
    "HIV_CD4_SYMPTOMATIC",
    "canonical_dw_key",
    "get_default_disability_weights",
    "resolve_disability_weights",
    "resolve_disease_module",
    "classify_hiv_stage",
    "hiv_stage_disability_weight",
    "combine_disability_weights",
    "multimorbidity_scale",
    "adjust_total_yld_for_multimorbidity",
]


# Canonical keys → provisional average DW (0 = full health, 1 = death)
DEFAULT_GBD_DISABILITY_WEIGHTS: dict[str, float] = {
    "hiv": 0.078,
    "type2diabetes": 0.049,
    "hypertension": 0.049,
    "cardiovasculardiseases": 0.072,
    "chronickidneydisease": 0.104,
}

CORE_CEA_CONDITIONS = tuple(DEFAULT_GBD_DISABILITY_WEIGHTS.keys())

# Phase-1 multimorbidity conventions (Task C)
# Base: multiplicative (Hilderink / GBD-style independent comorbidity)
# Primary sensitivity: additive (historical MicrocostingAnalyzer behavior)
MULTIMORBIDITY_RULES = ("multiplicative", "additive", "maximum")
DEFAULT_MULTIMORBIDITY_RULE = "multiplicative"
SENSITIVITY_MULTIMORBIDITY_RULE = "additive"

# Map common spellings → canonical key
DW_ALIASES: dict[str, str] = {
    "hiv": "hiv",
    "HIV": "hiv",
    "t2d": "type2diabetes",
    "T2D": "type2diabetes",
    "type2diabetes": "type2diabetes",
    "Type2Diabetes": "type2diabetes",
    "hypertension": "hypertension",
    "Hypertension": "hypertension",
    "htn": "hypertension",
    "HTN": "hypertension",
    "cardiovasculardiseases": "cardiovasculardiseases",
    "CardiovascularDiseases": "cardiovasculardiseases",
    "cvd": "cardiovasculardiseases",
    "CVD": "cardiovasculardiseases",
    "chronickidneydisease": "chronickidneydisease",
    "ChronicKidneyDisease": "chronickidneydisease",
    "ckd": "chronickidneydisease",
    "CKD": "chronickidneydisease",
}

# Provenance for Methods tables (not used at runtime)
DW_METADATA: dict[str, dict[str, str]] = {
    "hiv": {
        "gbd_health_state": "HIV/AIDS: receiving antiretroviral treatment",
        "source": "IHME GBD 2021 disability weights (Y2024M05D13)",
        "notes": "Average / single-DW sensitivity; Paper 1 base uses HIV_STAGE_DISABILITY_WEIGHTS",
    },
    "type2diabetes": {
        "gbd_health_state": "Diabetes mellitus: uncomplicated",
        "source": "Salomon et al. Lancet Glob Health 2015 (GBD 2013 DWs)",
        "notes": "Complications (neuropathy, amputation, etc.) parked with severity",
    },
    "hypertension": {
        "gbd_health_state": "Generic uncomplicated disease: worry and daily medication",
        "source": "Salomon et al. Lancet Glob Health 2015 (GBD 2013 DWs)",
        "notes": "Proxy — GBD treats high SBP mainly as a risk factor; provisional",
    },
    "cardiovasculardiseases": {
        "gbd_health_state": "Heart failure: moderate",
        "source": "Salomon et al. Lancet Glob Health 2015 (GBD 2013 DWs)",
        "notes": "Composite CVD module proxy until sequela mix is specified",
    },
    "chronickidneydisease": {
        "gbd_health_state": "Chronic kidney disease: stage IV",
        "source": "Salomon et al. Lancet Glob Health 2015 (GBD 2013 DWs)",
        "notes": "Mid/late-stage proxy; dialysis/ESRD not default",
    },
}

# ---------------------------------------------------------------------------
# Paper 1 Option A: STI-Sim HIV stage / ART → GBD 2021 sequela DWs
# Source file (CEA repo): data/IHME_GBD_2021_DISABILITY_WEIGHTS_Y2024M05D13.csv
# Prefer "without anemia" rows for the base map.
# ---------------------------------------------------------------------------
HIV_YLD_MODES = ("average", "stage")
DEFAULT_HIV_YLD_MODE = "average"  # MicrocostingAnalyzer default; Paper 1 overrides to stage

HIV_STAGE_DISABILITY_WEIGHTS: dict[str, float] = {
    "art": 0.078017329,  # HIV/AIDS with antiretroviral treatment without anemia
    "aids": 0.5815900900000004,  # AIDS without anemia
    "symptomatic": 0.27447870490000065,  # Symptomatic HIV without anemia
    "early": 0.012440830799999992,  # Early HIV without anemia
}

HIV_STAGE_METADATA: dict[str, dict[str, str]] = {
    "art": {
        "sequela": "HIV/AIDS with antiretroviral treatment without anemia",
        "health_state": "HIV/AIDS cases, receiving ARV treatment",
        "sti_sim_rule": "on_art",
    },
    "aids": {
        "sequela": "AIDS without anemia",
        "health_state": "AIDS cases, not receiving ARV treatment",
        "sti_sim_rule": "infected & ~on_art & cd4 < 200",
    },
    "symptomatic": {
        "sequela": "Symptomatic HIV without anemia",
        "health_state": "HIV cases, symptomatic, pre-AIDS",
        "sti_sim_rule": "falling OR (200 <= cd4 < 350) while untreated",
    },
    "early": {
        "sequela": "Early HIV without anemia",
        "health_state": "Generic uncomplicated disease: anxiety about diagnosis",
        "sti_sim_rule": "acute OR latent OR untreated with cd4 >= 350",
    },
}

# CD4 cutpoints aligned with STI-Sim mortality bins / AIDS definition
HIV_CD4_AIDS = 200.0
HIV_CD4_SYMPTOMATIC = 350.0


def canonical_dw_key(name: str) -> str:
    """Return lowercase canonical DW key for a condition name or alias."""
    if name in DW_ALIASES:
        return DW_ALIASES[name]
    lowered = name.lower()
    if lowered in DEFAULT_GBD_DISABILITY_WEIGHTS:
        return lowered
    if lowered in DW_ALIASES:
        return DW_ALIASES[lowered]
    return lowered


def get_default_disability_weights(
    conditions: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Return a copy of default GBD DWs.

    Parameters
    ----------
    conditions :
        Optional list of condition names/aliases. If given, only those keys
        (canonicalized) are returned. Unknown names are skipped.
    """
    if conditions is None:
        return deepcopy(DEFAULT_GBD_DISABILITY_WEIGHTS)

    out: dict[str, float] = {}
    for name in conditions:
        key = canonical_dw_key(name)
        if key in DEFAULT_GBD_DISABILITY_WEIGHTS:
            out[key] = DEFAULT_GBD_DISABILITY_WEIGHTS[key]
    return out


def resolve_disability_weights(
    disability_weights: Optional[Mapping[str, float]] = None,
    *,
    use_defaults: bool = True,
) -> dict[str, float]:
    """
    Build the DW dict used by MicrocostingAnalyzer.

    - ``None`` + ``use_defaults=True`` → core GBD defaults
    - explicit mapping → that mapping only (keys canonicalized where known)
    """
    if disability_weights is None:
        return get_default_disability_weights() if use_defaults else {}

    out: dict[str, float] = {}
    for name, weight in disability_weights.items():
        out[canonical_dw_key(name)] = float(weight)
    return out


def resolve_disease_module(diseases, cond: str):
    """
    Find a disease module on ``sim.diseases`` for a DW / condition key.

    Tries the raw key, lowercase, PascalCase aliases, and DW alias targets.
    Returns ``(module_or_None, resolved_name_or_None)``.
    """
    if diseases is None:
        return None, None

    candidates = []
    for c in (
        cond,
        cond.lower(),
        cond.lower().replace("_", ""),
        DW_ALIASES.get(cond),
        DW_ALIASES.get(cond.lower()),
        canonical_dw_key(cond),
    ):
        if c and c not in candidates:
            candidates.append(c)

    # Also try matching disease.name / disease_name on all modules
    for key in candidates:
        if hasattr(diseases, key):
            return getattr(diseases, key), key
        get = getattr(diseases, "get", None)
        if callable(get):
            mod = get(key, None)
            if mod is not None:
                return mod, key

    # Fallback: scan modules for matching name attributes
    target = canonical_dw_key(cond)
    values = getattr(diseases, "values", None)
    iterable = values() if callable(values) else []
    for mod in iterable:
        names = {
            str(getattr(mod, "name", "") or "").lower(),
            str(getattr(mod, "disease_name", "") or "").lower(),
            str(getattr(mod, "label", "") or "").lower(),
        }
        if target in names or cond.lower() in names:
            return mod, getattr(mod, "name", target)

    return None, None


def classify_hiv_stage(
    *,
    infected: np.ndarray,
    on_art: np.ndarray,
    cd4: np.ndarray,
    acute: Optional[np.ndarray] = None,
    latent: Optional[np.ndarray] = None,
    falling: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map STI-Sim HIV states to Option A stage labels (object array of str / '').

    Priority (first match wins): art → aids → symptomatic → early.
    """
    infected = np.asarray(infected, dtype=bool)
    on_art = np.asarray(on_art, dtype=bool)
    cd4 = np.asarray(cd4, dtype=float)
    n = infected.size
    stage = np.full(n, "", dtype=object)

    if not infected.any():
        return stage

    art = infected & on_art
    stage[art] = "art"

    untreated = infected & ~on_art
    if untreated.any():
        cd4_u = cd4.copy()
        # Missing CD4 among untreated → treat as early (conservative)
        missing = untreated & ~np.isfinite(cd4_u)
        cd4_u[missing] = HIV_CD4_SYMPTOMATIC + 1.0

        aids = untreated & (cd4_u < HIV_CD4_AIDS)
        stage[aids] = "aids"

        rem = untreated & (stage == "")
        falling_flag = np.zeros(n, dtype=bool)
        if falling is not None:
            falling_flag = np.asarray(falling, dtype=bool) & rem
        symptomatic = rem & (falling_flag | ((cd4_u >= HIV_CD4_AIDS) & (cd4_u < HIV_CD4_SYMPTOMATIC)))
        stage[symptomatic] = "symptomatic"

        rem = untreated & (stage == "")
        # acute / latent / high CD4 → early
        stage[rem] = "early"

    return stage


def hiv_stage_disability_weight(stage_labels: np.ndarray) -> np.ndarray:
    """Vector of DWs for ``classify_hiv_stage`` labels (0 if not infected / blank)."""
    out = np.zeros(len(stage_labels), dtype=float)
    for key, weight in HIV_STAGE_DISABILITY_WEIGHTS.items():
        out[stage_labels == key] = float(weight)
    return out


def combine_disability_weights(
    weights: Sequence[float] | Iterable[float],
    rule: str = DEFAULT_MULTIMORBIDITY_RULE,
) -> float:
    """
    Combine condition-level DWs into one comorbidity DW.

    Rules
    -----
    multiplicative : ``1 - ∏(1 - w_i)`` (base case; GBD / Hilderink-style)
    additive       : ``min(∑ w_i, 1)`` (sensitivity; can overstate joint disability)
    maximum        : ``max(w_i)`` (alternate sensitivity)
    """
    ws = [float(w) for w in weights if w is not None and float(w) > 0.0]
    if not ws:
        return 0.0

    rule = (rule or DEFAULT_MULTIMORBIDITY_RULE).lower()
    if rule not in MULTIMORBIDITY_RULES:
        raise ValueError(
            f"Unknown multimorbidity rule {rule!r}; choose from {MULTIMORBIDITY_RULES}"
        )

    if rule == "additive":
        return float(min(sum(ws), 1.0))
    if rule == "maximum":
        return float(max(ws))
    # multiplicative
    prod = 1.0
    for w in ws:
        prod *= 1.0 - min(max(w, 0.0), 1.0)
    return float(1.0 - prod)


def multimorbidity_scale(
    weights: Sequence[float] | Iterable[float],
    rule: str = DEFAULT_MULTIMORBIDITY_RULE,
) -> float:
    """
    Scale factor to convert additive duration×w YLD into a chosen comorbidity rule.

    ``adjusted_yld = additive_yld * scale``, with
    ``scale = combine(ws) / sum(ws)`` (1.0 if only one positive weight).

    Per-condition YLD columns stay additive for attribution; only totals adjust.
    """
    ws = [float(w) for w in weights if w is not None and float(w) > 0.0]
    if len(ws) <= 1:
        return 1.0
    denom = sum(ws)
    if denom <= 0:
        return 1.0
    return combine_disability_weights(ws, rule=rule) / denom


def adjust_total_yld_for_multimorbidity(
    yld_by_condition: Mapping[str, np.ndarray],
    disability_weights: Mapping[str, float],
    rule: str = DEFAULT_MULTIMORBIDITY_RULE,
) -> np.ndarray:
    """
    Build per-agent ``total_yld`` under a multimorbidity rule.

    ``yld_by_condition`` maps condition key → per-agent YLD array (already
    duration × weight × discount). Condition keys should match
    ``disability_weights`` (aliases are canonicalized).
    """
    if not yld_by_condition:
        return np.zeros(0, dtype=float)

    keys = list(yld_by_condition.keys())
    arrays = [np.asarray(yld_by_condition[k], dtype=float) for k in keys]
    n = len(arrays[0])
    additive = np.zeros(n, dtype=float)
    for arr in arrays:
        if len(arr) != n:
            raise ValueError("All condition YLD arrays must have the same length")
        additive += arr

    rule = (rule or DEFAULT_MULTIMORBIDITY_RULE).lower()
    if rule == "additive":
        return additive

    ws = np.array(
        [
            float(
                disability_weights.get(
                    canonical_dw_key(k), disability_weights.get(k, 0.0)
                )
            )
            for k in keys
        ],
        dtype=float,
    )
    present = np.column_stack([(arr > 0) for arr in arrays])  # (n, k)

    adjusted = additive.copy()
    multi_mask = present.sum(axis=1) > 1
    if not np.any(multi_mask):
        return adjusted

    for i in np.where(multi_mask)[0]:
        active_w = ws[present[i]]
        scale = multimorbidity_scale(active_w, rule=rule)
        adjusted[i] = additive[i] * scale
    return adjusted
