"""
Healthcare system access SDoH module.
"""

from .core import BaseSDoH

__all__ = ["HealthCareSystem"]


class HealthCareSystem(BaseSDoH):
    """Healthcare access / coverage."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="healthcare",
            csv_path=csv_path,
            condition_name="HealthCareSystem",
            default_p_stable=0.8,
            default_inherit_prob=0.9,
            state_attr="healthcare_system",
            **kwargs,
        )
        