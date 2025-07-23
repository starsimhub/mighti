"""
Defines interdependencies and risk modifiers between diseases and conditions.
"""


import pandas as pd
import matplotlib.pyplot as plt
import starsim as ss
import sciris as sc
from collections import defaultdict


class NCDHIVConnector(ss.Connector):
    """
    Connector to model interaction between HIV and NCDs by adjusting susceptibility.

    This connector increases the susceptibility to specified NCDs among HIV-infected agents
    by a user-specified relative susceptibility factor. It also tracks and optionally plots
    the dynamics of NCD prevalence, HIV prevalence, and susceptibility over time.

    Attributes:
        rel_sus_dict (dict): Dictionary mapping NCD names (str) to relative susceptibility values (float).
        time (sc.autolist): List of simulation times for plotting.
        rel_sus (defaultdict): Time-series of mean relative susceptibility for each NCD.
        ncd_prev (defaultdict): Time-series of NCD prevalence values.
        hiv_prev (sc.autolist): Time-series of HIV prevalence values.
    """

    def __init__(self, rel_sus_dict, pars=None, **kwargs):

        super().__init__(label='NCD-HIV')
        self.rel_sus_dict = rel_sus_dict
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

                self.rel_sus[ncd].append(ncd_obj.rel_sus.mean())
                self.ncd_prev[ncd].append(ncd_obj.results.prevalence[self.sim.ti])
                
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
            ax[i].plot(self.time, self.hiv_prev, label='HIV prevalence')
            ax[i].legend()
            ax[i].set_title(f'{self.sim.label} - {ncd}')
        
        plt.tight_layout()
        plt.show()
        return fig


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


def create_connectors(rel_sus):
    connectors = []
    for condition1, interactions in rel_sus.items():
        for condition2, rel_sus_val in interactions.items():
            connector = create_dynamic_connector(condition1, condition2, rel_sus_val)
            connectors.append(connector)
    return connectors


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
        return
        
    # Determine if the first condition uses 'infected' or 'affected'
        if hasattr(condition1_obj, 'infected'):
            condition1_uids = condition1_obj.infected.uids
        elif hasattr(condition1_obj, 'affected'):
            condition1_uids = condition1_obj.affected.uids
        else:
            raise AttributeError(f"{self.condition1} does not have 'infected' or 'affected' attribute.")
        
        # Apply the susceptibility adjustment to condition2 based on condition1
        condition2_obj.rel_sus[condition1_uids] = self.relative_risk
        return

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
