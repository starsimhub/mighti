"""
MIGHTI package top-level.

Public API policy (v2):
- Prefer *namespaces* (e.g. `mighti.diseases`, `mighti.analyzers`) over exporting
  every class/function into the top-level `mighti` namespace.
- Keep plotting utilities behind explicit imports (do not import matplotlib on
  `import mighti`).
"""

from .util.version import __version__, __versiondate__, __license__

# Expose common namespaces (preferred usage: `mi.diseases.Foo`, `mi.analyzers.Bar`)
from . import diseases  # noqa: F401
from . import analyzers  # noqa: F401
from . import interventions  # noqa: F401
from . import interactions  # noqa: F401
from . import sdoh  # noqa: F401
from . import economics  # noqa: F401
from . import calibration  # noqa: F401
from . import people_extend  # noqa: F401
from . import life_expectancy  # noqa: F401
from .util import figpaths  # noqa: F401
from .util import plot_style  # noqa: F401
from . import mortality_competing  # noqa: F401
from . import mortality_additive  # noqa: F401
from . import stisim_competing  # noqa: F401

# Keep a small set of core utilities convenient at top-level
from .disease_definitions import (  # noqa: F401
    initialize_prevalence_data,
    age_sex_dependent_prevalence,
)

# Keep adherence primitives convenient (used broadly)
from .adherence import (  # noqa: F401
    AdherenceEngine,
    ARTAdherenceDisruptor,
    InterventionAdherenceDisruptor,
    AdherenceFromDepression,
    CASM_REL_FACTORS,
    SDOH_REL_FACTORS,
)

# NOTE: plotting is intentionally NOT imported here.
# Use: `from mighti.plot_functions import ...`

def __getattr__(name: str):
    """
    Back-compat attribute resolver.

    This allows older code to keep working with `mi.Type2Diabetes`,
    `mi.SurvivorshipAnalyzer`, etc., while the preferred v2 style is:
      - `mi.diseases.Type2Diabetes`
      - `mi.analyzers.SurvivorshipAnalyzer`
      - `mi.interactions.NCDHIVConnector`
      - `mi.sdoh.NeighbourhoodSituation`
    """
    for mod in (
        diseases,
        analyzers,
        interventions,
        interactions,
        sdoh,
        people_extend,
        life_expectancy,
        figpaths,
        plot_style,
        mortality_competing,
        mortality_additive,
        stisim_competing,
    ):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module 'mighti' has no attribute {name!r}")


__all__ = [
    "__version__",
    "__versiondate__",
    "__license__",
    # namespaces
    "diseases",
    "analyzers",
    "interventions",
    "interactions",
    "sdoh",
    "economics",
    "calibration",
    "people_extend",
    "life_expectancy",
    "figpaths",
    "plot_style",
    "mortality_competing",
    "mortality_additive",
    "stisim_competing",
    # core utilities
    "initialize_prevalence_data",
    "age_sex_dependent_prevalence",
    # adherence
    "AdherenceEngine",
    "ARTAdherenceDisruptor",
    "InterventionAdherenceDisruptor",
    "AdherenceFromDepression",
    "CASM_REL_FACTORS",
    "SDOH_REL_FACTORS",
]