"""
Full integration test for MIGHTI.

This test verifies that a complete MIGHTI simulation runs without error
using Eswatini input data, including:
- Initialization of all modules (people, networks, diseases, interactions)
- Execution of interventions and analyzers
- Generation of mortality and life expectancy outputs
"""

import starsim as ss
import stisim as sti
import sciris as sc
import pandas as pd
import numpy as np
import os
import sys
import mighti as mi


def test_full_mighti_simulation():

    thisdir = os.path.dirname(__file__)
    
    inityear = 2007
    endyear = 2009
    n_agents = 500  # reduce for speed
    
    # Load parameters
    param_path = os.path.join(thisdir, 'test_data', 'eswatini_parameters.csv')
    df = pd.read_csv(param_path)
    df.columns = df.columns.str.strip()
    healthconditions = [cond for cond in df.condition if cond != "HIV"]
    diseases = ["HIV"] + healthconditions
    
    # Load prevalence data
    prev_path = os.path.join(thisdir, 'test_data', 'eswatini_prevalence.csv')
    prevalence_data_df = pd.read_csv(prev_path)
    prevalence_data, age_bins = mi.initialize_prevalence_data(healthconditions + ["HIV"], prevalence_data_df, inityear)
    
    # Initialize diseases
    def get_prev_fn(d):
        return lambda mod, sim, size: mi.age_sex_dependent_prevalence(d, prevalence_data, age_bins, sim, size)
    
    hiv = sti.HIV(init_prev=ss.bernoulli(get_prev_fn('HIV')),
                  beta={'structuredsexual': [0.01, 0.01], 'maternal': [0.01, 0.01]},
                  include_aids_deaths=True)
    
    disease_objects = [hiv]
    for d in healthconditions:
        cls = getattr(mi, d, None)
        if cls:
            disease_objects.append(cls(csv_path=param_path,
                                       pars={"init_prev": ss.bernoulli(get_prev_fn(d))}))
    
    # Load interactions
    rel_sus_path = os.path.join(thisdir, 'test_data', 'rel_sus.csv')
    df_rr = pd.read_csv(rel_sus_path)
    interactions = [mi.NCDHIVConnector(df.set_index('condition')['rel_sus'].to_dict())]
    
    # Load demographics
    death_path = os.path.join(thisdir, 'test_data', 'eswatini_mortality_rates_2007.csv')
    fertility_path = os.path.join(thisdir, 'test_data', 'eswatini_asfr.csv')
    age_path = os.path.join(thisdir, 'test_data', 'eswatini_age_distribution_2007.csv')
    
    deaths = ss.Deaths({'death_rate': pd.read_csv(death_path), 'rate_units': 1})
    pregnancy = ss.Pregnancy({'fertility_rate': pd.read_csv(fertility_path)})
    ppl = ss.People(n_agents, age_data=pd.read_csv(age_path))
    networks = [ss.MaternalNet(), sti.StructuredSexual()]

    # Analyzers
    analyzers = [
        mi.DeathsByAgeSexAnalyzer(),
        mi.SurvivorshipAnalyzer(),
        mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
    ]

    # Interventions
    interventions = [
        sti.HIVTest(test_prob_data=[0.6, 0.7, 0.95], years=[2000, 2007, 2016]),
        sti.ART(pars={'future_coverage': {'year': 2005, 'prop': 0.95}}),
        sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
        sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]})
    ]

    sim = ss.Sim(
        n_agents=n_agents,
        people=ppl,
        start=inityear,
        stop=endyear,
        demographics=[pregnancy, deaths],
        networks=networks,
        diseases=disease_objects,
        connectors=interactions,
        analyzers=analyzers,
        interventions=interventions,
        label="FullSimTest"
    )

    sim.run()

    # Minimal assertion: simulation completed
    assert sim.t.yearvec[-1] == endyear
    assert len(sim.people) >0
    assert abs(len(sim.people) - n_agents) < 10, "Too many agents dropped during setup."
    for analyzer in analyzers:
        assert hasattr(analyzer, 'results')
        
        
# Run as a script (optional)
if __name__ == '__main__':        
    test_full_mighti_simulation()
    