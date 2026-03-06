"""
Social context / support SDoH module.
"""

from .core import BaseSDoH

__all__ = ["SocialContext"]


class SocialContext(BaseSDoH):
    """Social support / connectedness."""
    def __init__(self, csv_path=None, **kwargs):
        super().__init__(
            name="social",
            csv_path=csv_path,
            condition_name="SocialContext",
            default_p_stable=0.75,
            default_inherit_prob=0.85,
            state_attr="social_context",
            **kwargs,
        )
        