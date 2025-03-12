import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict

# Specify all externally visible classes this file defines
__all__ = ['hiv_type2diabetes']

# Base class for HIV-related connectors
class HIVConnector(ss.Connector):
    """ Base class for connectors that increase susceptibility due to HIV """
    def __init__(self, label, requires, susceptibility_key, default_susceptibility, pars=None, **kwargs):
        super().__init__(label=label)
        self.define_pars(**{susceptibility_key: default_susceptibility})
        self.update_pars(pars, **kwargs)
        self.susceptibility_key = susceptibility_key
        self.requires = requires

    def step(self):
        sim = self.sim
        disease_name = self.susceptibility_key.split('_')[-1]
        disease_obj = getattr(sim.diseases, disease_name.lower())
        hiv_infected_uids = sim.people.hiv.infected.uids
        disease_obj.rel_sus[hiv_infected_uids] = self.pars[self.susceptibility_key]
        return

class hiv_type2diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type2Diabetes', [ss.HIV, mi.Type2Diabetes], 'rel_sus_hiv_type2diabetes', 1.5, pars, **kwargs)


# Functions to read in datafiles
def read_interactions(datafile=None):
    """
    Read in datafile with risk/condition interactions.
    At some point, this can be adjusted to automatically create the Connectors.
    """
    if datafile is None:
        datafile = sc.thispath() / '../mighti/data/rel_sus.csv'
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


class GenericNCDConnector(ss.Connector):
    """
    A generic connector to model interactions between two diseases.
    This class adjusts susceptibility based on relative risk from the interaction matrix.
    """

    def __init__(self, condition1, condition2, relative_risk, pars=None, **kwargs):
        """
        Initialize the connector with the two interacting conditions and their relative risk.
        """
        label = f'{condition1}-{condition2}'  # Create a unique label for each connector
        super().__init__(label=label, requires=[getattr(mi, condition1), getattr(mi, condition2)])
        self.condition1 = condition1
        self.condition2 = condition2
        self.relative_risk = relative_risk
        self.define_pars(rel_sus=relative_risk)  # Use the passed relative risk
        self.update_pars(pars, **kwargs)
        return

    def step(self):
        sim = self.sim
        cond1_obj = getattr(sim.diseases, self.condition1.lower())  # Get the first disease object
        cond2_obj = getattr(sim.diseases, self.condition2.lower())  # Get the second disease object
    
        # Determine if the second condition uses 'infected' or 'affected' and adjust accordingly
        if hasattr(cond1_obj, 'infected'):
            # If the first disease uses 'infected'
            condition1_uids = cond1_obj.infected.uids
        elif hasattr(cond1_obj, 'affected'):
            # If the first disease uses 'affected'
            condition1_uids = cond1_obj.affected.uids
        else:
            raise AttributeError(f"{self.condition1} does not have 'infected' or 'affected' attribute.")
        
        # Now, apply the susceptibility adjustment to condition2 based on condition1
        if hasattr(cond2_obj, 'infected'):
            # If the second disease uses 'infected'
            sim.diseases[self.condition2.lower()].rel_sus[condition1_uids] = self.pars.rel_sus
        elif hasattr(cond2_obj, 'affected'):
            # If the second disease uses 'affected'
            sim.diseases[self.condition2.lower()].rel_sus[condition1_uids] = self.pars.rel_sus
        else:
            raise AttributeError(f"{self.condition2} does not have 'infected' or 'affected' attribute.")
        return
