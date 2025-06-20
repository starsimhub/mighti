"""
Module defining HIV associated dementia as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease



class HIVAssociatedDementia(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'HIVAssociatedDementia'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'HIVAssociatedDementia'
        )
        return

