"""
Module defining breast cancer as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease



class BreastCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'BreastCancer'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'BreastCancer'
        )
        return

