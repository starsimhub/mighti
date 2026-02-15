"""
Neighbourhood / housing stability SDoH module.
"""

from __future__ import annotations

from .core import BaseSDoH

__all__ = ["NeighbourhoodSituation"]


class NeighbourhoodSituation(BaseSDoH):
    """Housing stability / neighbourhood situation (binary)."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="neighbourhood",
            csv_path=csv_path,
            condition_name="NeighbourhoodSituation",
            default_p_stable=0.98,  # 98% housed
            default_inherit_prob=0.9,
            state_attr="neighbourhood_situation",
            **kwargs,
        )
        # logger.info(f"[NeighbourhoodSituation] Module initialized with p_stable={self.p_stable:.3f}")
