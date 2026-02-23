"""
Module defining Influenza as an infectious disease model.
"""


from mighti.diseases.base_disease import GenericSIS



class Influenza(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Influenza'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'Influenza')
        return

