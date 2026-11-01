"""
MIGHTI: Model of Inter-Generational Health, Transmission, and Interventions

Use the namespaces, e.g. `mi.diseases.Type2Diabetes`, `mi.analyzers.PrevalenceAnalyzer_HIV`,
`mi.interactions.NCDHIVConnector`, `mi.sdoh.NeighbourhoodSituation`, `mi.interventions.ARTwithCASM`.
Plotting is not imported with MIGHTI: `from mighti.analysis.plotting import plot_mean_prevalence`.
"""

from .util.version import __version__, __versiondate__, __license__

from . import diseases
from . import analyzers
from . import analysis
from . import demography
from . import interventions
from . import interactions
from . import sdoh
from . import economics
from . import initialization

from .demography import people_extend, mortality_competing, mortality_additive
from .analysis import life_expectancy
from .util import figpaths, plot_style

# Used by most scripts
from .initialization.prevalence import initialize_prevalence_data, age_sex_dependent_prevalence
from .demography.people_extend import make_people_with_age_sex
from .interventions.adherence import (
    AdherenceEngine,
    ARTAdherenceDisruptor,
    InterventionAdherenceDisruptor,
    CASM_REL_FACTORS,
    SDOH_REL_FACTORS,
)
