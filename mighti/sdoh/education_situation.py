"""
Education situation SDoH module.
"""

from .core import BaseSDoH

__all__ = ["EducationSituation"]


class EducationSituation(BaseSDoH):
    """Educational attainment / stability."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="education",
            csv_path=csv_path,
            condition_name="EducationSituation",
            default_p_stable=0.65,
            default_inherit_prob=0.9,
            state_attr="education_situation",
            **kwargs,
        )
        