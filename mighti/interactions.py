import starsim as ss
import sciris as sc
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# People with HIV has higher risk of Type2Diabetes
class Type2DiabetesHIVConnector(ss.Connector):
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='Type2Diabetes-HIV')
        self.define_pars(rel_sus=1.95)
        self.update_pars(pars, **kwargs)
        self.time = sc.autolist()
        self.rel_sus = sc.autolist()
        self.t2d_prev = sc.autolist()
        self.hiv_prev = sc.autolist()
        
    def step(self):
        t2d = self.sim.diseases.type2diabetes
        hiv = self.sim.diseases.hiv
        t2d.rel_sus[hiv.infected.uids] = self.pars.rel_sus
        
        # Collecting data for analysis
        self.time += self.sim.t
        self.rel_sus += t2d.rel_sus.mean()
        self.t2d_prev += t2d.results.prevalence[self.sim.ti]
        self.hiv_prev += hiv.results.prevalence[self.sim.ti]
        return
    
    def plot(self):
        sc.options(dpi=200)
        fig = plt.figure()
        for key in ['rel_sus', 't2d_prev', 'hiv_prev']:
            plt.plot(self.time, self[key], label=key)
        plt.legend()
        plt.title(self.sim.label)
        plt.show()
        return fig
    
    
# People with HIV have a higher risk of Chronic Kidney Disease
class CKDHIVConnector(ss.Connector):
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='CKD-HIV')
        self.define_pars(rel_sus=2.0)  # Higher relative susceptibility for CKD in people with HIV
        self.update_pars(pars, **kwargs)
        self.time = sc.autolist()
        self.rel_sus = sc.autolist()
        self.ckd_prev = sc.autolist()
        self.hiv_prev = sc.autolist()
        
    def step(self):
        ckd = self.sim.diseases.chronickidneydisease
        hiv = self.sim.diseases.hiv
        ckd.rel_sus[hiv.infected.uids] = self.pars.rel_sus
        
        # Collecting data for analysis
        self.time += self.sim.t
        self.rel_sus += ckd.rel_sus.mean()
        self.ckd_prev += ckd.results.prevalence[self.sim.ti]
        self.hiv_prev += hiv.results.prevalence[self.sim.ti]
        return
    
    def plot(self):
        sc.options(dpi=200)
        fig = plt.figure()
        for key in ['rel_sus', 'ckd_prev'], 'hiv_prev':
            plt.plot(self.time, self[key], label=key)
        plt.legend()
        plt.title(self.sim.label)
        plt.show()
        return fig
    

# Function to read interaction data
def read_interactions(datafile=None):
    """
    Reads interaction data from a CSV file.
    Automatically creates the Connectors based on relative risk.
    """
    if datafile is None:
        datafile = '../mighti/data/rel_sus.csv'
    df = pd.read_csv(datafile, index_col=0)

    rel_sus = defaultdict(dict)

    for condition1 in df.index:
        for condition2 in df.columns:
            if condition1 != condition2:
                value = df.at[condition1, condition2]
                if not pd.isna(value):
                    rel_sus[condition1][condition2] = value

    return rel_sus

# Factory function to create connector classes
def create_connector_class(condition1, condition2, rel_sus_val):
    class DynamicConnector(ss.Connector):
        def __init__(self, pars=None, **kwargs):
            super().__init__(label=f'{condition1}-{condition2}')
            self.define_pars(rel_sus=rel_sus_val)
            self.update_pars(pars, **kwargs)
            self.time = sc.autolist()
            self.rel_sus = sc.autolist()
            self.condition1_prev = sc.autolist()
            self.condition2_prev = sc.autolist()

        def step(self):
            condition1_obj = self.sim.diseases[condition1.lower()]
            condition2_obj = self.sim.diseases[condition2.lower()]
            condition2_obj.rel_sus[condition1_obj.affected.uids] = self.pars.rel_sus

            # Collecting data for analysis
            self.time += self.sim.t
            self.rel_sus += condition2_obj.rel_sus.mean()
            self.condition1_prev += condition1_obj.results.prevalence[self.sim.ti]
            self.condition2_prev += condition2_obj.results.prevalence[self.sim.ti]

        def plot(self):
            sc.options(dpi=200)
            fig = plt.figure()
            for key in ['rel_sus', 'condition1_prev', 'condition2_prev']:
                plt.plot(self.time, self[key], label=key)
            plt.legend()
            plt.title(self.sim.label)
            plt.show()
            return fig

    return DynamicConnector

# Function to create connectors dynamically
def create_connectors(rel_sus):
    connectors = []
    for condition1, interactions in rel_sus.items():
        for condition2, rel_sus_val in interactions.items():
            connector_class = create_connector_class(condition1, condition2, rel_sus_val)
            connectors.append(connector_class())
    return connectors