from collections import defaultdict
import pandas as pd
import mighti as mi
import starsim as ss
import numpy as np
import sciris as sc

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
        disease_name = self.susceptibility_key.split('_')[-1].lower()
        
        if not hasattr(sim.diseases, disease_name):
            print(f"[ERROR] Disease {disease_name} not found in simulation. Skipping step.")
            return

        disease_obj = getattr(sim.diseases, disease_name)
        hiv_infected_uids = sim.people.hiv.infected.uids
        all_agents = np.arange(len(disease_obj.rel_sus))
        non_hiv_agents = np.setdiff1d(all_agents, hiv_infected_uids)

        # Reset rel_sus for non-HIV agents
        disease_obj.rel_sus.set(non_hiv_agents, 1.0)
    
        # Apply modified susceptibility to HIV-infected agents
        disease_obj.rel_sus.set(hiv_infected_uids, self.pars[self.susceptibility_key])


class GeneralHIVConnector(HIVConnector):
    """ Generalized class for HIV interactions with various NCDs. """
    def __init__(self, ncd_name, rel_sus_param, default_value, pars=None, **kwargs):
        if not hasattr(mi, ncd_name):
            print(f"[ERROR] NCD {ncd_name} not found in mighti. Skipping connector.")
            return
        
        disease_obj = getattr(mi, ncd_name)  
        super().__init__(f'HIV-{ncd_name}', [ss.HIV, disease_obj], rel_sus_param, default_value, pars, **kwargs)


def create_hiv_connectors(datafile=None):
    """
    Automatically generate HIV-NCD connectors from a parameter CSV file.
    """
    if datafile is None:
        datafile = sc.thispath() / '../mighti/data/rel_sus.csv'
    
    df = pd.read_csv(datafile)
    connectors = []

    for _, row in df.iterrows():
        condition = row["has_condition"]
        param_name = f"rel_sus_hiv_{condition.lower()}"
        default_value = row.get("HIV", None)  # Assume column "HIV" contains relative risk values

        if default_value is not None:
            connector = GeneralHIVConnector(condition, param_name, default_value)
            if connector is not None:
                connectors.append(connector)

    return connectors
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

#             # ✅ Check if the connector already exists
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

#         # ✅ Assign a unique name for each interaction
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