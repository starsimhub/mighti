import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict

# Specify all externally visible classes this file defines
__all__ = [
    'hiv_hypertension', 'hiv_obesity', 'hiv_type1diabetes', 'hiv_type2diabetes',
    'hiv_cardiovasculardiseases', 'hiv_chronickidneydisease', 'hiv_hyperlipidemia',
    'hiv_cervicalcancer', 'hiv_colorectalcancer', 'hiv_breastcancer', 'hiv_lungcancer',
    'hiv_prostatecancer', 'hiv_alcoholusedisorder', 'hiv_tobaccouse',
    'hiv_hivassociateddementia', 'hiv_ptsd', 'hiv_depression', 'hiv_hpv', 'hiv_flu',
    'hiv_viralhepatitis', 'hiv_domesticviolence', 'hiv_roadinjuries',
    'hiv_chronicliverdisease', 'hiv_asthma', 'hiv_copd', 'hiv_alzheimersdisease',
    'hiv_parkinsonsdisease', 'GenericNCDConnector', 'read_interactions'
]

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
        # print(f"Step called for {self.label}")  # Debugging

        sim = self.sim
        disease_name = self.susceptibility_key.split('_')[-1]
        disease_obj = getattr(sim.diseases, disease_name.lower(), None)

        if disease_obj is None:
            print(f"[ERROR] Disease {disease_name} not found in sim.diseases")  
            return

        if not hasattr(sim.people, 'hiv') or not hasattr(sim.people.hiv, 'infected'):
            print("[ERROR] sim.people.hiv.infected is missing")  
            return

        hiv_infected_uids = sim.people.hiv.infected.uids

        if not hasattr(disease_obj, 'rel_sus'):
            print(f"[ERROR] {disease_name} does not have 'rel_sus' attribute.")
            return

        # # Print only the first N individuals
        # N = 5  # Change this number as needed
        
        # for uid in list(hiv_infected_uids)[:N]:
        #     old_value = disease_obj.rel_sus[uid]
        #     print(f"ID {uid}: Before {old_value}, Condition: {self.label}")
        # print(f"Final rel_sus for T2D before incidence calculation: {disease_obj.rel_sus}")
        # Apply susceptibility update
        disease_obj.rel_sus[hiv_infected_uids] = self.pars[self.susceptibility_key]
        # print(f"Final rel_sus for T2D after incidence calculation: {disease_obj.rel_sus}")
        # # Print only the first N individuals after update
        # for uid in list(hiv_infected_uids)[:N]:
        #     new_value = disease_obj.rel_sus[uid]
        return
    
class hiv_type2diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type2Diabetes', [ss.HIV, mi.Type2Diabetes], 'rel_sus_hiv_type2diabetes', 20, pars, **kwargs)
    
    def step(self):        
        sim = self.sim
        disease_name = self.susceptibility_key.split('_')[-1]
        disease_obj = getattr(sim.diseases, disease_name.lower(), None)
    
        if disease_obj is None:
            print(f"[ERROR] Disease {disease_name} not found in sim.diseases")
            return
    
        hiv_infected_uids = sim.people.hiv.infected.uids
    
        if len(hiv_infected_uids) == 0:
            print("No HIV-infected individuals found.")
            return
        
        # print(f"HIV-infected UIDs: {list(hiv_infected_uids)[:5]}")  # Print first 5 for sanity check
    
        # Print before update
        # print(f"Before update, rel_sus (first 5 values): {disease_obj.rel_sus[hiv_infected_uids[:5]]}")
    
        # Apply update
        disease_obj.rel_sus.set(hiv_infected_uids, self.pars[self.susceptibility_key])
    
        # Print after update
        # print(f"After update, rel_sus (first 5 values): {disease_obj.rel_sus[hiv_infected_uids[:5]]}")
         

class hiv_type1diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type1Diabetes', [ss.HIV, mi.Type1Diabetes], 'rel_sus_hiv_type1diabetes', 1.95, pars, **kwargs)

class hiv_hypertension(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hypertension', [ss.HIV, mi.Hypertension], 'rel_sus_hiv_hypertension', 1.3, pars, **kwargs)

class hiv_hyperlipidemia(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hyperlipidemia', [ss.HIV, mi.Hyperlipidemia], 'rel_sus_hiv_hyperlipidemia', 1.3, pars, **kwargs)

class hiv_cardiovasculardiseases(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-CardiovascularDiseases', [ss.HIV, mi.CardiovascularDiseases], 'rel_sus_hiv_cardiovasculardiseases', 1.2, pars, **kwargs)


class hiv_asthma(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Asthma', [ss.HIV, mi.Asthma], 'rel_sus_hiv_asthma', 1.3, pars, **kwargs)

class hiv_copd(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-COPD', [ss.HIV, mi.COPD], 'rel_sus_hiv_copd', 1.3, pars, **kwargs)

class hiv_obesity(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Obesity', [ss.HIV, mi.Obesity], 'rel_sus_hiv_obesity', 1.2, pars, **kwargs)

class hiv_depression(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Depression', [ss.HIV, mi.Depression], 'rel_sus_hiv_depression', 2, pars, **kwargs)

class hiv_ptsd(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-PTSD', [ss.HIV, mi.PTSD], 'rel_sus_hiv_ptsd', 2, pars, **kwargs)

class hiv_hivassociateddementia(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-HIVAssociatedDementia', [ss.HIV, mi.HIVAssociatedDementia], 'rel_sus_hiv_hivassociateddementia', 2, pars, **kwargs)

class hiv_roadinjuries(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-RoadInjuries', [ss.HIV, mi.RoadInjuries], 'rel_sus_hiv_roadinjuries', 1.1, pars, **kwargs)

class hiv_domesticviolence(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-DomesticViolence', [ss.HIV, mi.DomesticViolence], 'rel_sus_hiv_domesticviolence', 1.1, pars, **kwargs)

class hiv_alzheimersdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-AlzheimersDisease', [ss.HIV, mi.AlzheimersDisease], 'rel_sus_hiv_alzheimersdisease', 1.1, pars, **kwargs)

class hiv_chronicliverdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ChronicLiverDisease', [ss.HIV, mi.ChronicLiverDisease], 'rel_sus_hiv_chronicliverdisease', 1.2, pars, **kwargs)

class hiv_heartdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Heart', [ss.HIV, mi.HeartDisease], 'rel_sus_hiv_heart', 1.3, pars, **kwargs)

class hiv_chronickidneydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ChronicKidneyDisease', [ss.HIV, mi.ChronicKidneyDisease], 'rel_sus_hiv_chronickidneydisease', 1.3, pars, **kwargs)

class hiv_flu(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Flu', [ss.HIV, mi.Flu], 'rel_sus_hiv_flu', 1.2, pars, **kwargs)

class hiv_hpv(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-HPV', [ss.HIV, mi.HPV], 'rel_sus_hiv_hpv', 1.5, pars, **kwargs)

class hiv_parkinsonsdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ParkinsonsDisease', [ss.HIV, mi.ParkinsonsDisease], 'rel_sus_hiv_parkinsonsdisease', 1.3, pars, **kwargs)

class hiv_tobaccouse(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-TobaccoUse', [ss.HIV, mi.TobaccoUse], 'rel_sus_hiv_tobaccouse', 1.5, pars, **kwargs)

class hiv_alcoholusedisorder(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-AlcoholUseDisorder', [ss.HIV, mi.AlcoholUseDisorder], 'rel_sus_hiv_alcoholusedisorder', 1.4, pars, **kwargs)

class hiv_cervicalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Cervical', [ss.HIV, mi.CervicalCancer], 'rel_sus_hiv_cervicalcancer', 1.3, pars, **kwargs)

class hiv_colorectalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Colorectal', [ss.HIV, mi.ColorectalCancer], 'rel_sus_hiv_colorectalcancer', 1.3, pars, **kwargs)

class hiv_breastcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Breast', [ss.HIV, mi.BreastCancer], 'rel_sus_hiv_breastcancer', 1.3, pars, **kwargs)

class hiv_lungcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Lung', [ss.HIV, mi.LungCancer], 'rel_sus_hiv_lungcancer', 1.4, pars, **kwargs)

class hiv_prostatecancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Prostate', [ss.HIV, mi.ProstateCancer], 'rel_sus_hiv_prostatecancer', 1.3, pars, **kwargs)

class hiv_viralhepatitis(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hepatitis', [ss.HIV, mi.ViralHepatitis], 'rel_sus_hiv_viralhepatitis', 1.3, pars, **kwargs)



# Generic connector for NCD-NCD interactions
class GenericNCDConnector(ss.Connector):
    """
    A generic connector to model interactions between two diseases.
    This class adjusts susceptibility based on relative risk from the interaction matrix.
    """

    def __init__(self, condition1, condition2, relative_risk, pars=None, **kwargs):
        label = f'{condition1}-{condition2}'#'-{unique_id}'  
        name = label.lower().replace(" ", "_")
        super().__init__(name=name, label=label)
        
        self.condition1 = condition1
        self.condition2 = condition2
        self.relative_risk = relative_risk
        self.define_pars(rel_sus=relative_risk)
        self.update_pars(pars, **kwargs)

    def step(self):
        sim = self.sim
        cond1_obj = getattr(sim.diseases, self.condition1.lower(), None)
        cond2_obj = getattr(sim.diseases, self.condition2.lower(), None)

        if not cond1_obj or not cond2_obj:
            print(f"[ERROR] {self.condition1} or {self.condition2} not found in sim.diseases")
            return

        condition1_uids = getattr(cond1_obj, "infected", None) or getattr(cond1_obj, "affected", None)
        if condition1_uids is None:
            print(f"[ERROR] {self.condition1} does not have 'infected' or 'affected' attribute.")
            return

        if hasattr(cond2_obj, "rel_sus"):
            cond2_obj.rel_sus[condition1_uids.uids] *= self.relative_risk
        else:
            print(f"[ERROR] {self.condition2} does not have 'rel_sus' attribute.")

        return


# Functions to read in datafiles
def read_interactions(datafile=None):
    """
    Read in datafile with risk/condition interactions.
    This can be adjusted to automatically create the Connectors.
    """
    if datafile is None:
        datafile = sc.thispath() / '../mighti/data/rel_sus.csv'
    df = pd.read_csv(datafile, index_col=0)

    rel_sus = defaultdict(dict)

    for condition1 in df.index:
        for condition2 in df.columns:
            if condition1 != condition2:
                value = df.at[condition1, condition2]
                if not pd.isna(value):
                    rel_sus[condition1][condition2] = value

    return rel_sus
    
    # from collections import defaultdict
# import pandas as pd
# import mighti as mi
# import starsim as ss
# import numpy as np
# import sciris as sc

# class HIVConnector(ss.Connector):
#     """ Base class for connectors that increase susceptibility due to HIV """
#     def __init__(self, label, requires, susceptibility_key, default_susceptibility, pars=None, **kwargs):
#         super().__init__(label=label)
#         self.define_pars(**{susceptibility_key: default_susceptibility})
#         self.update_pars(pars, **kwargs)
#         self.susceptibility_key = susceptibility_key
#         self.requires = requires

#     def step(self):
#         sim = self.sim  
#         disease_name = self.susceptibility_key.split('_')[-1].lower()
        
#         if not hasattr(sim.diseases, disease_name):
#             print(f"[ERROR] Disease {disease_name} not found in simulation. Skipping step.")
#             return

#         disease_obj = getattr(sim.diseases, disease_name)
#         hiv_infected_uids = sim.people.hiv.infected.uids
#         all_agents = np.arange(len(disease_obj.rel_sus))
#         non_hiv_agents = np.setdiff1d(all_agents, hiv_infected_uids)

#         # Reset rel_sus for non-HIV agents
#         disease_obj.rel_sus.set(non_hiv_agents, 1.0)
    
#         # Apply modified susceptibility to HIV-infected agents
#         disease_obj.rel_sus.set(hiv_infected_uids, self.pars[self.susceptibility_key])


# class GeneralHIVConnector(HIVConnector):
#     """ Generalized class for HIV interactions with various NCDs. """
#     def __init__(self, ncd_name, rel_sus_param, default_value, pars=None, **kwargs):
#         if not hasattr(mi, ncd_name):
#             print(f"[ERROR] NCD {ncd_name} not found in mighti. Skipping connector.")
#             return
        
#         disease_obj = getattr(mi, ncd_name)  
#         super().__init__(f'HIV-{ncd_name}', [ss.HIV, disease_obj], rel_sus_param, default_value, pars, **kwargs)


# def create_hiv_connectors(datafile=None):
#     """
#     Automatically generate HIV-NCD connectors from a parameter CSV file.
#     """
#     if datafile is None:
#         datafile = sc.thispath() / '../mighti/data/rel_sus.csv'
    
#     df = pd.read_csv(datafile)
#     connectors = []

#     for _, row in df.iterrows():
#         condition = row["has_condition"]
#         param_name = f"rel_sus_hiv_{condition.lower()}"
#         default_value = row.get("HIV", None)  # Assume column "HIV" contains relative risk values

#         if default_value is not None:
#             connector = GeneralHIVConnector(condition, param_name, default_value)
#             if connector is not None:
#                 connectors.append(connector)

#     return connectors
# # Functions to read in datafiles
# def read_ncd_ncd_interactions(datafile=None):
#     """
#     Reads in CSV file with NCD-NCD interactions and returns a dictionary 
#     mapping diseases to their interacting conditions.
#     """
#     if datafile is None:
#         datafile = sc.thispath() / '../mighti/data/rel_sus.csv'

#     df = pd.read_csv(datafile)

#     rel_sus = defaultdict(dict)

#     for _, row in df.iterrows():
#         condition1 = row['has_condition']
#         for condition2, value in row.items():
#             if condition2 == 'has_condition' or pd.isna(value):
#                 continue
#             rel_sus[condition1][condition2] = float(value)  # Convert to float for safety

#     return rel_sus

# def create_ncd_connectors():
#     """
#     Reads NCD-NCD interaction data and dynamically creates NCD connectors.
#     """
#     rel_sus_data = read_ncd_ncd_interactions()  # Read interaction data
#     connectors = {}

#     for condition1, interactions in rel_sus_data.items():
#         for condition2, relative_risk in interactions.items():
#             label = f"ncd_{condition1.lower()}_{condition2.lower()}"

#             # Check if the connector already exists
#             if label in connectors:
#                 print(f"[WARNING] Skipping duplicate connector: {label}")
#                 continue

#             connector = GenericNCDConnector(condition1, condition2, relative_risk)
#             connectors[label] = connector  # Store uniquely

#     return list(connectors.values())  # Convert dictionary to list

# class GenericNCDConnector(ss.Connector):
#     """
#     A generic connector to model interactions between two diseases.
#     This class adjusts susceptibility based on relative risk from the interaction matrix.
#     """

#     def __init__(self, condition1, condition2, relative_risk, pars=None, **kwargs):
#         """
#         Initialize the connector with the two interacting conditions and their relative risk.
#         """
#         if not hasattr(mi, condition1) or not hasattr(mi, condition2):
#             raise ValueError(f"Invalid disease names: {condition1}, {condition2}")

#         # Assign a unique name for each interaction
#         label = f"ncd_{condition1.lower()}_{condition2.lower()}"

#         super().__init__(label=label)
#         self.requires = [getattr(mi, condition1), getattr(mi, condition2)]

#         self.condition1 = condition1
#         self.condition2 = condition2
#         self.relative_risk = relative_risk
#         self.define_pars(rel_sus=relative_risk)
#         self.update_pars(pars, **kwargs)
        
#     def step(self):
#         sim = self.sim
#         cond1_obj = getattr(sim.diseases, self.condition1.lower())  # Get the first disease object
#         cond2_obj = getattr(sim.diseases, self.condition2.lower())  # Get the second disease object

#         if not cond1_obj or not cond2_obj:
#             print(f"[WARNING] {self.condition1} or {self.condition2} missing in sim. Skipping step.")
#             return

#         # Determine if the first condition uses 'infected' or 'affected'
#         if hasattr(cond1_obj, 'infected'):
#             condition1_uids = cond1_obj.infected.uids
#         elif hasattr(cond1_obj, 'affected'):
#             condition1_uids = cond1_obj.affected.uids
#         else:
#             raise AttributeError(f"{self.condition1} does not have 'infected' or 'affected' attribute.")

#         # Apply susceptibility adjustments to condition2
#         if hasattr(cond2_obj, 'infected'):
#             cond2_obj.rel_sus[condition1_uids] = self.pars.rel_sus
#         elif hasattr(cond2_obj, 'affected'):
#             cond2_obj.rel_sus[condition1_uids] = self.pars.rel_sus
#         else:
#             raise AttributeError(f"{self.condition2} does not have 'infected' or 'affected' attribute.")