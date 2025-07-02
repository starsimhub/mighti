"""
Module defining the Type 2 Diabetes remitting disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class Type2Diabetes(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type2Diabetes'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label='Type2Diabetes')  
        if not hasattr(self.pars, 'p_acquire_multiplier'):
            self.pars.p_acquire_multiplier = 0.092
        return

