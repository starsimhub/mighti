"""
MIGHTI social determinants of health (SDoH) package.

Preferred usage:
    - `import mighti as mi`
    - `sdoh = mi.sdoh.NeighbourhoodSituation(csv_path=...)`

This initializer keeps imports lightweight by lazily loading submodules/classes.
"""

import importlib


_SUBMODULES = {
    "core",
    "neighbourhood_situation",
    "social_context",
    "education_situation",
    "economic_situation",
    "healthcare_system",
}

_EXPORTS = {
    # Base
    "BaseSDoH": "core",
    # Concrete modules
    "NeighbourhoodSituation": "neighbourhood_situation",
    "SocialContext": "social_context",
    "EducationSituation": "education_situation",
    "EconomicSituation": "economic_situation",
    "HealthCareSystem": "healthcare_system",
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

