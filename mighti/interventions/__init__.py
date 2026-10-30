"""
MIGHTI interventions package.

Preferred usage:
    - `import mighti as mi`
    - `intv = mi.interventions.ART(...)`

This initializer keeps imports lightweight by lazily loading submodules/classes.
"""

import importlib


_SUBMODULES = {
    "core",
    "adherence",
    "ncd_treatment",
}

_EXPORTS = {
    # interventions/core.py
    "ART": "core",
    "ARTwithCASM": "core",
    "ARTNoAutoAdjust": "core",
    "ImproveHospitalDischarge": "core",
    "GiveHousingToDepressed": "core",
    "GiveHousingSupport": "core",
    "HousingSupportForAUD": "core",
    # interventions/adherence.py
    "AdherenceEngine": "adherence",
    "ARTAdherenceDisruptor": "adherence",
    "InterventionAdherenceDisruptor": "adherence",
    "AdherenceFromDepression": "adherence",
    "CASM_REL_FACTORS": "adherence",
    "SDOH_REL_FACTORS": "adherence",
    "BASELINE_ADHERENCE_PHARMACOTHERAPY": "adherence",
    "CASM_NONADHERENCE_OR": "adherence",
    # interventions/ncd_treatment.py
    "ChronicCareTx": "ncd_treatment",
    "ReduceMortalityTx": "ncd_treatment",
    "make_ncd_treatment": "ncd_treatment",
    "chronic_care_eligible_mask": "ncd_treatment",
    "ELIGIBILITY_MODES": "ncd_treatment",
}

__all__ = sorted([*_SUBMODULES, *_EXPORTS.keys()])


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

