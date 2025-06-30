"""
Module defining the Type 2 Diabetes remitting disease model.
"""


import numpy as np
import starsim as ss
from mighti.diseases.base_disease import RemittingDisease


class Type2Diabetes(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type2Diabetes'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label='Type2Diabetes')  
        if not hasattr(self.pars, 'p_acquire_multiplier'):
            self.pars.p_acquire_multiplier = 0.092
        return

