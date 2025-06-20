"""
Module defining cardiovascular disease as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class CardiovascularDiseases(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'CardiovascularDiseases'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'CardiovascularDiseases'
        )
        return

