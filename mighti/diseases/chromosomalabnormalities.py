"""
Module defining chromosomal abnormalities as a static condition (lifelong, non-remitting).
"""

from mighti.diseases.base_disease import StaticCondition


class ChromosomalAbnormalities(StaticCondition):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ChromosomalAbnormalities'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='ChromosomalAbnormalities')
        return
    