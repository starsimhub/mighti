"""
Module defining road injuries as an acute disease model.
"""


from mighti.diseases.base_disease import AcuteDisease


class RoadInjuries(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'RoadInjuries'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'RoadInjuries')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

