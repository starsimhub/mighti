import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict
import numpy as np

# Specify all externally visible classes this file defines
__all__ = [
    'hiv_hypertension', 'hiv_obesity', 'hiv_type1diabetes', 'hiv_type2diabetes',
    'hiv_depression', 'hiv_accident', 'hiv_alzheimers', 'hiv_assault',
    'hiv_cerebrovasculardisease', 'hiv_chronicliverdisease',
    'hiv_chroniclowerrespiratorydisease', 'hiv_heartdisease',
    'hiv_chronickidneydisease', 'hiv_flu', 'hiv_hpv', 'hiv_parkinsons',
    'hiv_smoking', 'hiv_alcohol', 'hiv_brca', 'hiv_viralhepatitis',
    'hiv_cervicalcancer', 'hiv_colorectalcancer', 'hiv_breastcancer', 'hiv_lungcancer',
    'hiv_prostatecancer', 'hiv_othercancer', 'hiv_poverty',
    'GenericNCDConnector', 'read_interactions'
]


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
        all_agents = np.arange(len(disease_obj.rel_sus))
        non_hiv_agents = np.setdiff1d(all_agents, hiv_infected_uids)

        print(f"[DEBUG] Before update: {disease_name} rel_sus first 5 values: {disease_obj.rel_sus[:5]}")

        # Reset rel_sus for non-HIV agents
        disease_obj.rel_sus.set(non_hiv_agents, 1.0)

        # Ensure rel_sus is modified correctly
        print(f"[DEBUG] rel_sus ID before: {id(disease_obj.rel_sus)}")
        disease_obj.rel_sus.set(hiv_infected_uids, self.pars[self.susceptibility_key])
        print(f"[DEBUG] rel_sus ID after: {id(disease_obj.rel_sus)}")

        # Debug available disease attributes
        print(f"[DEBUG] {disease_name} attributes: {dir(disease_obj)}")

        # Check transmission rate location
        if hasattr(sim, "transmission_rate"):
            print(f"[DEBUG] Transmission rate found in sim: {sim.transmission_rate[:5]}")
        elif hasattr(disease_obj, "transmission_rate"):
            print(f"[DEBUG] Transmission rate found in {disease_name}: {disease_obj.transmission_rate[:5]}")
        else:
            print(f"[WARNING] Transmission rate not found in sim or disease.")

        print(f"[DEBUG] After update: {disease_name} rel_sus first 5 values: {disease_obj.rel_sus[:5]}")

        # Check if age_bins exist
        print(f"[DEBUG] age_bins: {getattr(sim, 'age_bins', 'Not Found')}")
        

        
# class hiv_type2diabetes(HIVConnector):
#     def __init__(self, pars=None, **kwargs):
#         super().__init__('HIV-Type2Diabetes', [ss.HIV, mi.Type2Diabetes], 'rel_sus_hiv_type2diabetes', 1.95, pars, **kwargs)
class hiv_type2diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type2Diabetes', [ss.HIV, mi.Type2Diabetes], 'rel_sus_hiv_type2diabetes', 1.95, pars, **kwargs)

    def step(self):
        sim = self.sim
        t2d_obj = getattr(sim.diseases, 'type2diabetes', None)

        if t2d_obj is None:
            print("[DEBUG] Type2Diabetes object not found!")
            return

        hiv_infected_uids = sim.people.hiv.infected.uids

        print(f"[DEBUG] Step {sim.ti}: Before update - rel_sus (first 5) {t2d_obj.rel_sus[:5]}")

        # Reset rel_sus for non-HIV agents
        all_agents = np.arange(len(t2d_obj.rel_sus))
        non_hiv_agents = np.setdiff1d(all_agents, hiv_infected_uids)
        t2d_obj.rel_sus.set(non_hiv_agents, 1.0)

        # Apply increased susceptibility for HIV-positive individuals
        t2d_obj.rel_sus.set(hiv_infected_uids, self.pars['rel_sus_hiv_type2diabetes'])

        print(f"[DEBUG] Step {sim.ti}: After update - rel_sus (first 5) {t2d_obj.rel_sus[:5]}")
        

class hiv_hypertension(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hypertension', [ss.HIV, mi.Hypertension], 'rel_sus_hiv_hypertension', 1.3, pars, **kwargs)

class hiv_obesity(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Obesity', [ss.HIV, mi.Obesity], 'rel_sus_hiv_obesity', 1.2, pars, **kwargs)

class hiv_type1diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type1Diabetes', [ss.HIV, mi.Type1Diabetes], 'rel_sus_hiv_type1diabetes', 1.5, pars, **kwargs)


class hiv_depression(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Depression', [ss.HIV, mi.Depression], 'rel_sus_hiv_depression', 2, pars, **kwargs)

class hiv_accident(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Accident', [ss.HIV, mi.Accident], 'rel_sus_hiv_accident', 1.1, pars, **kwargs)

class hiv_alzheimers(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Alzheimers', [ss.HIV, mi.Alzheimers], 'rel_sus_hiv_alzheimers', 1.1, pars, **kwargs)

class hiv_assault(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Assault', [ss.HIV, mi.Assault], 'rel_sus_hiv_assault', 1.1, pars, **kwargs)

class hiv_cerebrovasculardisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Cerebrovascular', [ss.HIV, mi.CerebrovascularDisease], 'rel_sus_hiv_cerebro', 1.2, pars, **kwargs)

class hiv_chronicliverdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Liver', [ss.HIV, mi.ChronicLiverDisease], 'rel_sus_hiv_liver', 1.2, pars, **kwargs)

class hiv_chroniclowerrespiratorydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Respiratory', [ss.HIV, mi.ChronicLowerRespiratoryDisease], 'rel_sus_hiv_resp', 1.2, pars, **kwargs)

class hiv_heartdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Heart', [ss.HIV, mi.HeartDisease], 'rel_sus_hiv_heart', 1.3, pars, **kwargs)

class hiv_chronickidneydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Kidney', [ss.HIV, mi.ChronicKidneyDisease], 'rel_sus_hiv_kidney', 1.3, pars, **kwargs)

class hiv_flu(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Flu', [ss.HIV, mi.Flu], 'rel_sus_hiv_flu', 1.2, pars, **kwargs)

class hiv_hpv(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-HPV', [ss.HIV, mi.HPV], 'rel_sus_hiv_hpv', 1.5, pars, **kwargs)

class hiv_parkinsons(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Parkinsons', [ss.HIV, mi.Parkinsons], 'rel_sus_hiv_parkinsons', 1.3, pars, **kwargs)

class hiv_smoking(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Smoking', [ss.HIV, mi.Smoking], 'rel_sus_hiv_smoking', 1.5, pars, **kwargs)

class hiv_alcohol(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Alcohol', [ss.HIV, mi.Alcohol], 'rel_sus_hiv_alcohol', 1.4, pars, **kwargs)

class hiv_brca(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-BRCA', [ss.HIV, mi.BRCA], 'rel_sus_hiv_brca', 1.3, pars, **kwargs)

class hiv_cervicalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Cervical', [ss.HIV, mi.CervicalCancer], 'rel_sus_hiv_cervical', 1.3, pars, **kwargs)

class hiv_colorectalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Colorectal', [ss.HIV, mi.ColorectalCancer], 'rel_sus_hiv_colorectal', 1.3, pars, **kwargs)

class hiv_breastcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Breast', [ss.HIV, mi.BreastCancer], 'rel_sus_hiv_breast', 1.3, pars, **kwargs)

class hiv_lungcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Lung', [ss.HIV, mi.LungCancer], 'rel_sus_hiv_lung', 1.4, pars, **kwargs)

class hiv_prostatecancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Prostate', [ss.HIV, mi.ProstateCancer], 'rel_sus_hiv_prostate', 1.3, pars, **kwargs)

class hiv_othercancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Other', [ss.HIV, mi.OtherCancer], 'rel_sus_hiv_other', 1.3, pars, **kwargs)

class hiv_viralhepatitis(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hepatitis', [ss.HIV, mi.ViralHepatitis], 'rel_sus_hiv_hepatitis', 1.3, pars, **kwargs)

class hiv_poverty(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Poverty', [ss.HIV, mi.Poverty], 'rel_sus_hiv_poverty', 1.3, pars, **kwargs)

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
    
    # import starsim as ss
# import mighti as mi
# import pandas as pd
# import sciris as sc

# # -------------------------
# # Read HIV Interactions
# # -------------------------

# df_interactions = None  # Placeholder for external data

# def initialize_interactions(data):
#     """ Function to initialize interactions with preloaded interaction data """
#     global df_interactions
#     df_interactions = data

# def read_hiv_interactions():
#     """ Read HIV interactions using preloaded data. """
#     rel_sus = {}
#     for _, row in df_interactions.iterrows():
#         condition = row['condition']
#         rel_sus_value = row['relative_risk']
#         rel_sus[condition] = rel_sus_value  # Store only HIV-related interactions
#     return rel_sus


# # -------------------------
# # Generic HIV-NCD Connector
# # -------------------------

# class HIVConnector(ss.Connector):
#     """
#     Generic connector to model increased susceptibility due to HIV.
#     """

#     def __init__(self, condition, relative_risk, pars=None, **kwargs):
#         label = f'HIV-{condition}'
#         super().__init__(label=label, requires=[ss.HIV, getattr(mi, condition, None)])

#         if None in self.requires:
#             raise ValueError(f"Condition {condition} not found in `mighti`.")

#         self.condition = condition
#         self.relative_risk = relative_risk
#         self.define_pars(rel_sus=relative_risk)
#         self.update_pars(pars, **kwargs)

#     def step(self):
#         sim = self.sim
#         condition_obj = getattr(sim.diseases, self.condition.lower(), None)

#         if not condition_obj:
#             return  # Skip if the disease object is not initialized

#         # Get HIV-infected individuals
#         hiv_infected_uids = sim.people.hiv.infected.uids

#         # Adjust relative susceptibility
#         setattr(condition_obj, 'rel_sus', {uid: self.pars.rel_sus for uid in hiv_infected_uids})
#         return


# # -------------------------
# # Create HIV-NCD Connectors Dynamically
# # -------------------------

# def create_hiv_connectors():
#     """
#     Reads HIV interaction data and dynamically creates HIV-NCD connectors.
#     """
#     rel_sus_data = read_hiv_interactions()
#     connectors = []

#     for condition, relative_risk in rel_sus_data.items():
#         connector_label = f'hiv_{condition.lower()}'

#         if connector_label not in globals():
#             # Dynamically create and add a new connector class
#             connector_class = type(
#                 connector_label,
#                 (HIVConnector,),
#                 {
#                     '__init__': lambda self, pars=None, **kwargs: super(connector_class, self).__init__(
#                         condition, relative_risk, pars, **kwargs
#                     )
#                 }
#             )

#             globals()[connector_label] = connector_class  # Register globally
#             connectors.append(connector_class())

#     return connectors