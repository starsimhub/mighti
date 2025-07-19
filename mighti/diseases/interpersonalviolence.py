"""
Module defining domestic violence as an acute disease model.
"""


from mighti.diseases.base_disease import AcuteDisease


class InterpersonalViolence(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'InterpersonalViolence'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'InterpersonalViolence'
        )
        return

