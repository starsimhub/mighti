import pandas as pd
import starsim as ss
import mighti as mi
from collections import defaultdict

# Define your custom connectors or interactions as needed
# class hiv_depression(ss.Connector):
#     """ Simple connector to make people with depression more likely to contract HIV """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Depression', requires=[ss.hiv, mi.Depression])
#         self.default_pars(
#             rel_sus_hiv_depression=2,  # People with depression are 2x more likely to acquire HIV
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with depression
#         sim.diseases.hiv.rel_sus[sim.people.depression.affected] = self.pars.rel_sus_hiv_depression
#         return

class hiv_depression(ss.Connector):
    """ Simple connector to make people with depression more likely to contract HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Depression', requires=[ss.HIV, mi.Depression])  # Use ss.HIV class
        self.default_pars(
            rel_sus_hiv_depression=2,  # People with depression are 2x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return
    
    def update(self):
        sim = self.sim
        # print(f"Depression affected: {sim.people.depression.affected}")  # Check affected individuals
        # print(f"Before update: {sim.diseases.hiv.rel_sus}")  # Check initial relative susceptibility
    
        # Apply the increased susceptibility to those with depression
        sim.diseases.hiv.rel_sus[sim.people.depression.affected] = self.pars.rel_sus_hiv_depression
    
        # print(f"After update: {sim.diseases.hiv.rel_sus}")  # Check updated susceptibility
        return

# Functions to read in datafiles
def read_interactions(datafile=None):
    """
    Read in datafile with risk/condition interactions
    Note, this is not yet used, but at some point this could be adjusted so that it automatically
    created the Connectors
    """
    if datafile is None:
        datafile = '../mighti/data/rel_sus.csv'
    df = pd.read_csv(datafile)

    rel_sus = defaultdict(dict)

    for cond in df.has_condition.unique():
        conddf = df.loc[df.has_condition == cond]
        conddf.reset_index(inplace=True, drop=True)
        interacting_conds = conddf.columns[~conddf.isna().any()].tolist()
        interacting_conds.remove('has_condition')
        for interacting_cond in interacting_conds:
            rel_sus[interacting_cond][cond] = conddf.loc[0, interacting_cond]

    return rel_sus

