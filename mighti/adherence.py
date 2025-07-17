"""
MIGHTI Adherence Module

Stores and updates agent-level adherence to health interventions,
modulated by CASM conditions and social determinants.
"""

import numpy as np
import starsim as ss
import sciris as sc


def create_adherence_connector(intervention_name):
    class_name = f"AdherenceConnector_{intervention_name.lower()}"
    
    class DynamicAdherenceConnector(ss.Connector):
        def __init__(self):
            label = f"adherence_{intervention_name.lower()}"
            super().__init__(label=label)
            self.intervention_name = intervention_name
            self.time = sc.autolist()
            self.mean_adherence = sc.autolist()

        def step(self):
            sim = self.sim
            ppl = sim.people
            uids = ppl.uid
            adherence_vals = np.ones(len(uids))
        
            conditions = sim.diseases
            sdoh = getattr(ppl, 'sdoh', {})
        
            if 'depression' in conditions:
                depressed = conditions['depression'].affected
                adherence_vals[depressed] *= 1 / 2.21
            if 'alcoholusedisorder' in conditions:
                drinkers = conditions['alcoholusedisorder'].affected
                adherence_vals[drinkers] *= 1 / 1.41
        
            adherence_vals = np.clip(adherence_vals, 0.0, 1.0)
            self.mean_adherence.append(adherence_vals.mean())
            self.time.append(sim.t)
        
            intv = next((i for i in sim.interventions
                         if hasattr(i, 'label') and i.label == self.intervention_name), None)        
            
            if hasattr(intv, 'rel_effect'):
                intv.rel_effect[uids] *= adherence_vals

    DynamicAdherenceConnector.__name__ = class_name
    return DynamicAdherenceConnector()


class DepressionTreatmentEffectConnector(ss.Connector):
    def __init__(self, target_intervention='ART', label=None):
        label = label or f'depression_tx_effect_on_{target_intervention.lower()}'
        super().__init__(label=label)
        self.target_intervention = target_intervention

    def step(self):
        sim = self.sim
        ppl = sim.people
        uids = ppl.uids
        boost = np.ones(len(uids))

        # Check who is receiving depression treatment
        for intv in sim.interventions:
            if 'depression' in intv.label.lower():
                if hasattr(intv, 'receiving') and isinstance(intv.receiving, np.ndarray):
                    boost[intv.receiving] *= 1.2  # Example: 20% adherence improvement

        # Apply to target intervention
        target = next((i for i in sim.interventions if i.label == self.target_intervention), None)
        if target and hasattr(target, 'rel_effect'):
            target.rel_effect[uids] *= np.clip(boost, 0, 1.5)
            