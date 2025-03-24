# Exact copy of minimal_mighti
import starsim as ss
import sciris as sc
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class NCDHIVConnector(ss.Connector):
    def __init__(self, rel_sus_dict, pars=None, **kwargs):
        super().__init__(label='NCD-HIV')
        self.rel_sus_dict = rel_sus_dict  # Dictionary containing relative susceptibility values for each NCD
        self.update_pars(pars, **kwargs)
        
        self.time = sc.autolist()
        self.rel_sus = defaultdict(sc.autolist)
        self.ncd_prev = defaultdict(sc.autolist)
        self.hiv_prev = sc.autolist()
        
    def step(self):
        hiv = self.sim.diseases.hiv
        
        for ncd, rel_sus_val in self.rel_sus_dict.items():
            ncd_obj = self.sim.diseases.get(ncd.lower(), None)
            if ncd_obj is not None:
                ncd_obj.rel_sus[hiv.infected.uids] = rel_sus_val

                # Collecting data for analysis
                self.rel_sus[ncd].append(ncd_obj.rel_sus.mean())
                self.ncd_prev[ncd].append(ncd_obj.results.prevalence[self.sim.ti])
                
                # print(f"Relative susceptibility for {ncd} due to HIV: {ncd_obj.rel_sus[hiv.infected.uids]}")

                
        self.time.append(self.sim.t)
        self.hiv_prev.append(hiv.results.prevalence[self.sim.ti])
        return
    
    def plot(self):
        sc.options(dpi=200)
        fig, ax = plt.subplots(len(self.rel_sus_dict), 1, figsize=(10, 8))
        
        if len(self.rel_sus_dict) == 1:
            ax = [ax]
        
        for i, ncd in enumerate(self.rel_sus_dict.keys()):
            ax[i].plot(self.time, self.rel_sus[ncd], label=f'{ncd} rel_sus')
            ax[i].plot(self.time, self.ncd_prev[ncd], label=f'{ncd} prevalence')
            ax[i].plot(self.time, self.hiv_prev, label=f'HIV prevalence')
            ax[i].legend()
            ax[i].set_title(f'{self.sim.label} - {ncd}')
        
        plt.tight_layout()
        plt.show()
        return fig


# Function to read interaction data
def read_interactions(datafile=None):
    """
    Reads interaction data from a CSV file.
    Automatically creates the Connectors based on relative risk.
    """
    if datafile is None:
        datafile = 'rel_sus.csv'
    df = pd.read_csv(datafile, index_col=0)

    rel_sus = defaultdict(dict)

    for condition1 in df.index:
        for condition2 in df.columns:
            if condition1 != condition2:
                value = df.at[condition1, condition2]
                if not pd.isna(value):
                    rel_sus[condition1][condition2] = value

    return rel_sus

# Function to create connectors for each pair of conditions
def create_connectors(rel_sus):
    connectors = []
    for condition1, interactions in rel_sus.items():
        for condition2, rel_sus_val in interactions.items():
            connector = create_dynamic_connector(condition1, condition2, rel_sus_val)
            connectors.append(connector)
    return connectors

# Function to create a dynamic connector with unique class name
def create_dynamic_connector(condition1, condition2, rel_sus_val):
    class_name = f"{condition1}_{condition2}_Connector"
    DynamicConnector = type(class_name, (ss.Connector,), {
        '__init__': lambda self, pars=None, **kwargs: super(DynamicConnector, self).__init__(label=f'{condition1}-{condition2}'),
        'step': lambda self: step_function(self, condition1, condition2, rel_sus_val),
        'plot': plot_function,
        'define_pars': lambda self, rel_sus=rel_sus_val: setattr(self, 'pars', sc.objdict(rel_sus=rel_sus)),
        'update_pars': ss.Connector.update_pars,
        'time': sc.autolist(),
        'rel_sus': sc.autolist(),
        'condition1_prev': sc.autolist(),
        'condition2_prev': sc.autolist()
    })
    return DynamicConnector()

def step_function(self, condition1, condition2, rel_sus_val):
    condition1_obj = self.sim.diseases.get(condition1.lower(), None)
    condition2_obj = self.sim.diseases.get(condition2.lower(), None)
    
    if condition1_obj is None or condition2_obj is None:
        # print(f"Error: {condition1} or {condition2} not found in simulation diseases.")
        return
        
    condition2_obj.rel_sus[condition1_obj.affected.uids] = rel_sus_val


    # print(f"Relative susceptibility for {condition2} due to {condition1}: {condition2_obj.rel_sus[condition1_obj.affected.uids]}")

    # Collecting data for analysis
    self.time.append(self.sim.t)
    self.rel_sus.append(condition2_obj.rel_sus.mean())
    self.condition1_prev.append(condition1_obj.results.prevalence[self.sim.ti])
    self.condition2_prev.append(condition2_obj.results.prevalence[self.sim.ti])



def plot_function(self):
    sc.options(dpi=200)
    fig = plt.figure()
    for key in ['rel_sus', 'condition1_prev', 'condition2_prev']:
        plt.plot(self.time, self[key], label=key)
    plt.legend()
    plt.title(self.sim.label)
    plt.show()
    return fig



# import starsim as ss
# import sciris as sc
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # People with HIV has higher risk of Type2Diabetes
# class Type2DiabetesHIVConnector(ss.Connector):
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='Type2Diabetes-HIV')
#         self.define_pars(rel_sus=1.95)
#         self.update_pars(pars, **kwargs)
#         self.time = sc.autolist()
#         self.rel_sus = sc.autolist()
#         self.t2d_prev = sc.autolist()
#         self.hiv_prev = sc.autolist()
        
#     def step(self):
#         t2d = self.sim.diseases.type2diabetes
#         hiv = self.sim.diseases.hiv
#         t2d.rel_sus[hiv.infected.uids] = self.pars.rel_sus
        
#         # Collecting data for analysis
#         self.time += self.sim.t
#         self.rel_sus += t2d.rel_sus.mean()
#         self.t2d_prev += t2d.results.prevalence[self.sim.ti]
#         self.hiv_prev += hiv.results.prevalence[self.sim.ti]
#         return
    
#     def plot(self):
#         sc.options(dpi=200)
#         fig = plt.figure()
#         for key in ['rel_sus', 't2d_prev', 'hiv_prev']:
#             plt.plot(self.time, self[key], label=key)
#         plt.legend()
#         plt.title(self.sim.label)
#         plt.show()
#         return fig
    
    
# # People with HIV have a higher risk of Chronic Kidney Disease
# class CKDHIVConnector(ss.Connector):
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='CKD-HIV')
#         self.define_pars(rel_sus=2.0)  # Higher relative susceptibility for CKD in people with HIV
#         self.update_pars(pars, **kwargs)
#         self.time = sc.autolist()
#         self.rel_sus = sc.autolist()
#         self.ckd_prev = sc.autolist()
#         self.hiv_prev = sc.autolist()
        
#     def step(self):
#         ckd = self.sim.diseases.chronickidneydisease
#         hiv = self.sim.diseases.hiv
#         ckd.rel_sus[hiv.infected.uids] = self.pars.rel_sus
        
#         # Collecting data for analysis
#         self.time += self.sim.t
#         self.rel_sus += ckd.rel_sus.mean()
#         self.ckd_prev += ckd.results.prevalence[self.sim.ti]
#         self.hiv_prev += hiv.results.prevalence[self.sim.ti]
#         return
    
#     def plot(self):
#         sc.options(dpi=200)
#         fig = plt.figure()
#         for key in ['rel_sus', 'ckd_prev', 'hiv_prev']:
#             plt.plot(self.time, self[key], label=key)
#         plt.legend()
#         plt.title(self.sim.label)
#         plt.show()
#         return fig


# # Function to read interaction data
# def read_interactions(datafile=None):
#     """
#     Reads interaction data from a CSV file.
#     Automatically creates the Connectors based on relative risk.
#     """
#     if datafile is None:
#         datafile = 'rel_sus.csv'
#     df = pd.read_csv(datafile, index_col=0)

#     rel_sus = defaultdict(dict)

#     for condition1 in df.index:
#         for condition2 in df.columns:
#             if condition1 != condition2:
#                 value = df.at[condition1, condition2]
#                 if not pd.isna(value):
#                     rel_sus[condition1][condition2] = value

#     return rel_sus

# # Function to create connectors for each pair of conditions
# def create_connectors(rel_sus):
#     connectors = []
#     for condition1, interactions in rel_sus.items():
#         for condition2, rel_sus_val in interactions.items():
#             connector = create_dynamic_connector(condition1, condition2, rel_sus_val)
#             connectors.append(connector)
#     return connectors

# # Function to create a dynamic connector with unique class name
# def create_dynamic_connector(condition1, condition2, rel_sus_val):
#     class_name = f"{condition1}_{condition2}_Connector"
#     DynamicConnector = type(class_name, (ss.Connector,), {
#         '__init__': lambda self, pars=None, **kwargs: super(DynamicConnector, self).__init__(label=f'{condition1}-{condition2}'),
#         'step': lambda self: step_function(self, condition1, condition2, rel_sus_val),
#         'plot': plot_function,
#         'define_pars': lambda self, rel_sus=rel_sus_val: setattr(self, 'pars', sc.objdict(rel_sus=rel_sus)),
#         'update_pars': ss.Connector.update_pars,
#         'time': sc.autolist(),
#         'rel_sus': sc.autolist(),
#         'condition1_prev': sc.autolist(),
#         'condition2_prev': sc.autolist()
#     })
#     return DynamicConnector()

# def step_function(self, condition1, condition2, rel_sus_val):
#     condition1_obj = self.sim.diseases.get(condition1.lower(), None)
#     condition2_obj = self.sim.diseases.get(condition2.lower(), None)
    
#     if condition1_obj is None or condition2_obj is None:
#         print(f"Error: {condition1} or {condition2} not found in simulation diseases.")
#         return
        
#     condition2_obj.rel_sus[condition1_obj.affected.uids] = rel_sus_val


#     # Collecting data for analysis
#     self.time += self.sim.t
#     self.rel_sus += condition2_obj.rel_sus.mean()
#     self.condition1_prev += condition1_obj.results.prevalence[self.sim.ti]
#     self.condition2_prev += condition2_obj.results.prevalence[self.sim.ti]



# def plot_function(self):
#     sc.options(dpi=200)
#     fig = plt.figure()
#     for key in ['rel_sus', 'condition1_prev', 'condition2_prev']:
#         plt.plot(self.time, self[key], label=key)
#     plt.legend()
#     plt.title(self.sim.label)
#     plt.show()
#     return fig