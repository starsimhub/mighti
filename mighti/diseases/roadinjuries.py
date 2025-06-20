"""
Module defining road injuries as an acute disease model.
"""


from mighti.diseases.base_disease import AcuteDisease


class RoadInjuries(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'RoadInjuries'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'RoadInjuries'
        )
        return

