"""
Module defining Down syndrome as a static condition (lifelong, non-remitting).
"""

from mighti.diseases.base_disease import StaticCondition


class DownSyndrome(StaticCondition):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DownSyndrome'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='DownSyndrome')
        return
    