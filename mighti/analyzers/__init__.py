"""
MIGHTI analyzers package.

Public API:
- Prefer the namespace style: `mi.analyzers.PrevalenceAnalyzer_HIV`, etc.
- Keep imports light by lazily loading submodules/classes on first access.
"""

import importlib


# Map public names -> module that defines them
_EXPORTS = {
    # analyzer_core.py
    "DeathsByAgeSexAnalyzer": "analyzer_core",
    "AgeSexMxAnalyzer": "analyzer_core",
    "SurvivorshipAnalyzer": "analyzer_core",
    "ConditionAtDeathAnalyzer": "analyzer_core",
    "CauseOfDeathYLLAnalyzer": "analyzer_core",
    # analyzer_cost.py
    "MicrocostingAnalyzer": "analyzer_cost",
    "HRHAnalyzer": "analyzer_cost",
    "summarize_microcosting_results": "analyzer_cost",
    # disability_weights.py
    "DEFAULT_GBD_DISABILITY_WEIGHTS": "disability_weights",
    "DW_METADATA": "disability_weights",
    "CORE_CEA_CONDITIONS": "disability_weights",
    "MULTIMORBIDITY_RULES": "disability_weights",
    "DEFAULT_MULTIMORBIDITY_RULE": "disability_weights",
    "SENSITIVITY_MULTIMORBIDITY_RULE": "disability_weights",
    "HIV_YLD_MODES": "disability_weights",
    "DEFAULT_HIV_YLD_MODE": "disability_weights",
    "HIV_STAGE_DISABILITY_WEIGHTS": "disability_weights",
    "HIV_STAGE_METADATA": "disability_weights",
    "get_default_disability_weights": "disability_weights",
    "resolve_disability_weights": "disability_weights",
    "resolve_disease_module": "disability_weights",
    "canonical_dw_key": "disability_weights",
    "classify_hiv_stage": "disability_weights",
    "hiv_stage_disability_weight": "disability_weights",
    "combine_disability_weights": "disability_weights",
    "multimorbidity_scale": "disability_weights",
    "adjust_total_yld_for_multimorbidity": "disability_weights",
    # analyzer_prevalence.py
    "PrevalenceAnalyzer": "analyzer_prevalence",
    "PrevalenceAnalyzer_HIV": "analyzer_prevalence",
    "PrevalenceAnalyzer_SDoH": "analyzer_prevalence",
    "CauseDeathRateAnalyzer": "analyzer_prevalence",
    "OnARTByConditionAnalyzer": "analyzer_prevalence",
    "OnARTByConditionAndSexAnalyzer": "analyzer_prevalence",
    # analyzer_intervention.py
    "InterventionAnalyzer": "analyzer_intervention",
    "AdherenceAnalyzer": "analyzer_intervention",
    # analyzer_serviceuse.py
    "HospitalizationAnalyzer": "analyzer_serviceuse",
    "OutpatientVisitAnalyzer": "analyzer_serviceuse",
    "PreventiveServiceAnalyzer": "analyzer_serviceuse",
    "ERVisitAnalyzer": "analyzer_serviceuse",
}

# Also allow `mighti.analyzers.analyzer_core` access without importing eagerly
_SUBMODULES = {
    "analyzer_core",
    "analyzer_cost",
    "analyzer_prevalence",
    "analyzer_intervention",
    "analyzer_serviceuse",
    "disability_weights",
}

__all__ = sorted([*_EXPORTS.keys(), *_SUBMODULES])


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")

    modname = _EXPORTS.get(name)
    if modname is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{modname}")
    return getattr(mod, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
    