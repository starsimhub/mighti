"""
Module defining the Chronic Pain remitting disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class ChronicPain(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ChronicPain'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'ChronicPain')

        return

