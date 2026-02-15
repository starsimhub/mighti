"""
Module defining COVID-19 as an infectious disease model.

This is intentionally a thin wrapper around `GenericSIS` so that COVID-19 can be
included/excluded in a model via the same parameter CSV mechanism as other
conditions.
"""

from mighti.diseases.base_disease import GenericSIS


class COVID19(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = "COVID19"
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label="COVID19")
        return

