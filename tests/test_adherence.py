import mighti as mi
import numpy as np
import pandas as pd
import starsim as ss
import stisim as sti
import sciris as sc
import os

from mighti.adherence import create_adherence_connector, DepressionTreatmentEffectConnector


# Settings
do_plot = False
sc.options(interactive=do_plot)

# File path to parameter file
thisdir = os.path.dirname(__file__)
age_path = os.path.join(thisdir, 'test_data', 'eswatini_age_distribution_2007.csv')  # simple flat age distribution
prev_path = os.path.join(thisdir,'test_data', 'eswatini_prevalence.csv')

def test_adherence_connector_runs():
    n = 1000
    seed = 42
    ss.set_seed(seed)


    # Manually set 10% prevalence for CASM conditions
    casm_conditions = ['MajorDepressiveDisorder','AlcoholUseDisorder','AnxietyDisorder',
                       'ChronicPain','TobaccoUse','OpioidUseDisorder','StimulantUseDisorder']
    
    
    prevalence_data_df = pd.read_csv(prev_path)
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        casm_conditions, prevalence_data=prevalence_data_df, inityear=2007
    )
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)
    
    disease_objects = []

    for disease in casm_conditions:
        disease_class = getattr(mi, disease, None)
        if disease_class:
            init_prev = ss.bernoulli(get_prevalence_function(disease))
            disease_obj = disease_class(csv_path=prev_path, pars={"init_prev": init_prev})
            disease_objects.append(disease_obj)
            
    # ART coverage among PLHIV (from 95-95-95 cascade estimates and Lancet data)
    art_coverage_data = pd.DataFrame({
        'p_art': [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]
    }, index=[2003, 2010, 2013, 2014, 2016, 2022])
    art = sti.ART(coverage_data=art_coverage_data)

    # Setup sim
    sim = ss.Sim(
        n_agents=n,
        start=2020,
        stop=2021,
        people=ss.People(n, age_data=pd.read_csv(age_path)),
        diseases=disease_objects,
        interventions=[art],
        connectors=[create_adherence_connector('ART')],
        label='Adherence Test'
    )
    
    sim.run()

    # Check that rel_effect has been reduced due to CASM
    adherence_val = art.rel_effect
    assert np.any(adherence_val < 1.0), "Adherence was not affected by CASM conditions"
    print("✅ AdherenceConnector reduced intervention effect as expected.")

def test_depression_treatment_boost():
    n = 1000
    ppl = ss.make_people(n_agents=n)
    ppl.initialize()

    # Setup dummy depression intervention with receiving array
    depression_tx = ss.Intervention(label='depression_tx')
    depression_tx.receiving = np.zeros(n, dtype=bool)
    depression_tx.receiving[:100] = True  # 10% getting treated

    # Dummy ART
    art = ss.Intervention(label='ART')
    art.rel_effect = np.ones(n)

    # Setup sim
    sim = ss.Sim(
        n_agents=n,
        start=2020,
        stop=2021,
        people=ppl,
        interventions=[art, depression_tx],
        connectors=[DepressionTreatmentEffectConnector()],
        label='Depression Boost Test'
    )

    sim.run()
    boosted = art.rel_effect
    assert boosted[:100].mean() > 1.0, "Boosted adherence not applied"
    print("✅ DepressionTreatmentEffectConnector boosted adherence for treated agents.")

if __name__ == '__main__':
    test_adherence_connector_runs()
    test_depression_treatment_boost()