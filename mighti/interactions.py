"""
Specify interactions between diseases, conditions, and risks
"""

import pandas as pd
import starsim as ss
import mighti as mi
from collections import defaultdict

# Specify all externally visible classes this file defines
__all__ = [
    'hiv_obesity','hiv_hypertension','hiv_type1diabetes','hiv_type2diabetes','hiv_depression',
    'hiv_accident', 'hiv_alzheimers', 'hiv_cerebro', 'hiv_liver', 'hiv_resp', 
    'hiv_heart', 'hiv_kidney', 'hiv_flu','hiv_hpv', 'hiv_other', 'hiv_parkinsons', 
    'hiv_smoking', 'hiv_alcohol', 'hiv_brca',
    'hiv_colorectal', 'hiv_breast', 'hiv_lung', 'hiv_prostate', 
    'read_interactions'
]


class hiv_hypertension(ss.Connector):
    """ Simple connector to make people with hypertension more likely to contract HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Hypertension', requires=[ss.HIV, mi.Hypertension])
        self.default_pars(
            rel_sus_hiv_hypertension=1.3,  # People with hypertension are 1.3x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with hypertension
        sim.diseases.hiv.rel_sus[sim.people.hypertension.affected] = self.pars.rel_sus_hiv_hypertension
        return


class hiv_obesity(ss.Connector):
    """ Simple connector to make people with obesity more likely to contract HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Obesity', requires=[ss.HIV, mi.Obesity])
        self.default_pars(
            rel_sus_hiv_obesity=1.4,  # People with obesity are 1.4x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with obesity
        sim.diseases.hiv.rel_sus[sim.people.obesity.affected] = self.pars.rel_sus_hiv_obesity
        return
    
# class hiv_diabetes(ss.Connector):
#     """ Simple connector to make people with diabetes more likely to contract HIV """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Diabetes', requires=[ss.HIV, mi.Diabetes])
#         self.default_pars(
#             rel_sus_hiv_diabetes=1.5,  # People with diabetes are 1.5x more likely to acquire HIV
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with diabetes
#         sim.diseases.hiv.rel_sus[sim.people.diabetes.affected] = self.pars.rel_sus_hiv_diabetes
#         return
    
class hiv_type1diabetes(ss.Connector):
    """ Simple connector to make people with diabetes more likely to contract HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Type1Diabetes', requires=[ss.HIV, mi.Type1Diabetes])
        self.default_pars(
            rel_sus_hiv_type1diabetes=1.5,  # People with diabetes are 1.5x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with diabetes
        sim.diseases.hiv.rel_sus[sim.people.type1diabetes.affected] = self.pars.rel_sus_hiv_type1diabetes
        return
    
    
class hiv_type2diabetes(ss.Connector):
    """ Simple connector to make people with diabetes more likely to contract HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Type2Diabetes', requires=[ss.HIV, mi.Type2Diabetes])
        self.default_pars(
            rel_sus_hiv_type2diabetes=1.5,  # People with diabetes are 1.5x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with diabetes
        sim.diseases.hiv.rel_sus[sim.people.type2diabetes.affected] = self.pars.rel_sus_hiv_type2diabetes
        return


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
        # Apply the increased susceptibility to those with depression
        sim.diseases.hiv.rel_sus[sim.people.depression.affected] = self.pars.rel_sus_hiv_depression
        return


class hiv_accident(ss.Connector):
    """ Connector for accidents increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Accident', requires=[ss.HIV])
        self.default_pars(rel_sus_hiv_accident=1.1)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply susceptibility change for accidents (replace with real logic)
        return


class hiv_alzheimers(ss.Connector):
    """ Connector for Alzheimer's increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Alzheimers', requires=[ss.HIV, mi.Alzheimers])
        self.default_pars(rel_sus_hiv_alzheimers=1.1)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.alzheimers.affected] = self.pars.rel_sus_hiv_alzheimers
        return


class hiv_cerebro(ss.Connector):
    """ Connector for cerebrovascular diseases increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Cerebrovascular', requires=[ss.HIV, mi.Cerebro])
        self.default_pars(rel_sus_hiv_cerebro=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.cerebro.affected] = self.pars.rel_sus_hiv_cerebro
        return


class hiv_liver(ss.Connector):
    """ Connector for liver diseases increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Liver', requires=[ss.HIV, mi.Liver])
        self.default_pars(rel_sus_hiv_liver=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.liver.affected] = self.pars.rel_sus_hiv_liver
        return


class hiv_resp(ss.Connector):
    """ Connector for respiratory diseases increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Respiratory', requires=[ss.HIV, mi.Resp])
        self.default_pars(rel_sus_hiv_resp=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.resp.affected] = self.pars.rel_sus_hiv_resp
        return


class hiv_heart(ss.Connector):
    """ Connector for heart diseases increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Heart', requires=[ss.HIV, mi.Heart])
        self.default_pars(rel_sus_hiv_heart=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.heart.affected] = self.pars.rel_sus_hiv_heart
        return


class hiv_kidney(ss.Connector):
    """ Connector for kidney diseases increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Kidney', requires=[ss.HIV, mi.Kidney])
        self.default_pars(rel_sus_hiv_kidney=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.kidney.affected] = self.pars.rel_sus_hiv_kidney
        return


class hiv_flu(ss.Connector):
    """ Connector for influenza increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Flu', requires=[ss.HIV, mi.Flu])
        self.default_pars(rel_sus_hiv_flu=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.flu.affected] = self.pars.rel_sus_hiv_flu
        return


class hiv_hpv(ss.Connector):
    """ Connector for HPV increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-HPV', requires=[ss.HIV, mi.HPV])
        self.default_pars(rel_sus_hiv_hpv=1.5)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.hpv.affected] = self.pars.rel_sus_hiv_hpv
        return


class hiv_other(ss.Connector):
    """ Generic connector for other diseases increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Other', requires=[ss.HIV, mi.Other])
        self.default_pars(rel_sus_hiv_other=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.other.affected] = self.pars.rel_sus_hiv_other
        return


class hiv_parkinsons(ss.Connector):
    """ Connector for Parkinson's disease increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Parkinsons', requires=[ss.HIV, mi.Parkinsons])
        self.default_pars(rel_sus_hiv_parkinsons=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.parkinsons.affected] = self.pars.rel_sus_hiv_parkinsons
        return


class hiv_smoking(ss.Connector):
    """ Connector for smoking increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Smoking', requires=[ss.HIV, mi.Smoking])
        self.default_pars(rel_sus_hiv_smoking=1.5)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.smoking.affected] = self.pars.rel_sus_hiv_smoking
        return


class hiv_alcohol(ss.Connector):
    """ Connector for alcohol use increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Alcohol', requires=[ss.HIV, mi.Alcohol])
        self.default_pars(rel_sus_hiv_alcohol=1.4)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.alcohol.affected] = self.pars.rel_sus_hiv_alcohol
        return


class hiv_brca(ss.Connector):
    """ Connector for BRCA mutation increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-BRCA', requires=[ss.HIV, mi.BRCA])
        self.default_pars(rel_sus_hiv_brca=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.brca.affected] = self.pars.rel
        return
    
class hiv_colorectal(ss.Connector):
    """ Connector for colorectal cancer increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Colorectal', requires=[ss.HIV, mi.Colorectal])
        self.default_pars(
            rel_sus_hiv_colorectal=1.3  # People with colorectal cancer are 1.3x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.colorectal.affected] = self.pars.rel_sus_hiv_colorectal
        return


class hiv_breast(ss.Connector):
    """ Connector for breast cancer increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Breast', requires=[ss.HIV, mi.Breast])
        self.default_pars(
            rel_sus_hiv_breast=1.3  # People with breast cancer are 1.3x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.breast.affected] = self.pars.rel_sus_hiv_breast
        return


class hiv_lung(ss.Connector):
    """ Connector for lung cancer increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Lung', requires=[ss.HIV, mi.Lung])
        self.default_pars(
            rel_sus_hiv_lung=1.4  # People with lung cancer are 1.4x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.lung.affected] = self.pars.rel_sus_hiv_lung
        return


class hiv_prostate(ss.Connector):
    """ Connector for prostate cancer increasing susceptibility to HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Prostate', requires=[ss.HIV, mi.Prostate])
        self.default_pars(
            rel_sus_hiv_prostate=1.3  # People with prostate cancer are 1.3x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.prostate.affected] = self.pars.rel_sus_hiv_prostate
        return 


    
# Functions to read in datafiles
def read_interactions(datafile=None):
    """
    Read in datafile with risk/condition interactions.
    At some point, this can be adjusted to automatically create the Connectors.
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