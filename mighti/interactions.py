import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc

# -------------------------
# Read HIV Interactions
# -------------------------

df_interactions = None  # Placeholder for external data

def initialize_interactions(data):
    """ Function to initialize interactions with preloaded interaction data """
    global df_interactions
    df_interactions = data

def read_hiv_interactions():
    """ Read HIV interactions using preloaded data. """
    rel_sus = {}
    for _, row in df_interactions.iterrows():
        condition = row['condition']
        rel_sus_value = row['relative_risk']
        rel_sus[condition] = rel_sus_value  # Store only HIV-related interactions
    return rel_sus


# -------------------------
# Generic HIV-NCD Connector
# -------------------------

class HIVConnector(ss.Connector):
    """
    Generic connector to model increased susceptibility due to HIV.
    """

    def __init__(self, condition, relative_risk, pars=None, **kwargs):
        label = f'HIV-{condition}'
        super().__init__(label=label, requires=[ss.HIV, getattr(mi, condition, None)])
        
        if None in self.requires:
            raise ValueError(f"Condition {condition} not found in `mighti`.")

        self.condition = condition
        self.relative_risk = relative_risk
        self.define_pars(rel_sus=relative_risk)
        self.update_pars(pars, **kwargs)

    def step(self):
        sim = self.sim
        condition_obj = getattr(sim.diseases, self.condition.lower(), None)

        if not condition_obj:
            return  # Skip if the disease object is not initialized

        # Get HIV-infected individuals
        hiv_infected_uids = sim.people.hiv.infected.uids

        # Adjust relative susceptibility
        setattr(condition_obj, 'rel_sus', {uid: self.pars.rel_sus for uid in hiv_infected_uids})
        return


# -------------------------
# Create HIV-NCD Connectors Dynamically
# -------------------------

def create_hiv_connectors():
    """
    Reads HIV interaction data and dynamically creates HIV-NCD connectors.
    """
    rel_sus_data = read_hiv_interactions()
    connectors = []

    for condition, relative_risk in rel_sus_data.items():
        connector_label = f'hiv_{condition.lower()}'

        if connector_label not in globals():
            # Dynamically create and add a new connector class
            connector_class = type(
                connector_label,
                (HIVConnector,),
                {
                    '__init__': lambda self, pars=None, **kwargs: super(connector_class, self).__init__(
                        condition, relative_risk, pars, **kwargs
                    )
                }
            )
            
            globals()[connector_label] = connector_class  # Register globally
            connectors.append(connector_class())

    return connectors
