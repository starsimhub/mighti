"""
Module defining Maternal conditions as an acute disease model.

This is a placeholder acute condition primarily used for mortality attribution
in competing-risks mode. In practice, you may want to restrict acquisition to
reproductive ages and pregnancy status; for now we rely on `affected_sex=female`
in the parameter CSV and any scenario-specific risk multipliers.
"""

from mighti.diseases.base_disease import AcuteDisease


class MaternalConditions(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = "MaternalConditions"
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label="MaternalConditions")
        return

