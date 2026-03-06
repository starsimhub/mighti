"""
Economic situation SDoH module.
"""

from .core import BaseSDoH

__all__ = ["EconomicSituation"]


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
        