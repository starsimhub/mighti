"""
Module defining asthma as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class Asthma(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Asthma'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'Asthma')
        if not hasattr(self.pars, 'p_acquire_multiplier'):
            self.pars.p_acquire_multiplier = 0.001
        return

