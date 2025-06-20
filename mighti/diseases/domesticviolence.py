"""
Module defining domestic violence as an acute disease model.
"""


from mighti.diseases.base_disease import AcuteDisease


class DomesticViolence(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DomesticViolence'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'DomesticViolence'
        )
        return

