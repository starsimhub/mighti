"""
Helpers for initializing People with custom sex ratio.

This module exists for compatibility with the functionality repository and
provides a `PeopleCustom` class that overrides the baseline female probability.
"""

from __future__ import annotations

import starsim as ss

__all__ = ["PeopleCustom"]


class PeopleCustom(ss.People):
    """
    StarSim People with a custom overall female probability.

    Notes
    -----
    - Keeps the module name 'people' so StarSim internals can plan correctly.
    - Override the BoolState default sampler for `female`.
    """

    def __init__(self, n_agents, age_data=None, extra_states=None, mock: bool = False):  # noqa: ARG002
        super().__init__(n_agents=n_agents, age_data=age_data, extra_states=extra_states)
        self.name = "people"

        # Override the default sampler used when .init_vals() runs
        self.female.default = ss.bernoulli(name="female", p=0.522)

