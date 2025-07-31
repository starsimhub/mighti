"""
Module defining the Chronic Pain remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease


class ChronicPain(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ChronicPain'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ChronicPain')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

