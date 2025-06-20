"""
Module defining prostate cancer as a remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease


class PTSD(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'PTSD'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'PTSD'
        )
        return

