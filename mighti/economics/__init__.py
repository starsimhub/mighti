"""
Economic evaluation subpackage (optional).

This subpackage is intentionally not imported into the core `mighti` namespace.
"""

# NOTE: BudgetConstraint and other economic utilities are intentionally not
# re-exported by default for the non-economic evaluation package build.
from .resource_accounting import *
from .utils import *