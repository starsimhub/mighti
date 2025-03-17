import pandas as pd
import starsim as ss
import mighti as mi

def run_simulation(prevalence_file, demographics_file, fertility_file, mortality_file, init_year, end_year, population_size):
    # Function to load CSV files
    def load_csv(file):
        return pd.read_csv(file)

    prevalence_data = load_csv(prevalence_file)
    demographics_data = load_csv(demographics_file)

    # Initialize prevalence data
    prevalence_data, age_bins = mi.initialize_prevalence_data(['HIV', 'Type2Diabetes', 'ChronicKidneyDisease'], prevalence_data, init_year)

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

    # Initialize the PrevalenceAnalyzer
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=['HIV', 'Type2Diabetes', 'ChronicKidneyDisease'])

    # Create demographics
    fertility_rates = {'fertility_rate': pd.read_csv(fertility_file)}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(mortality_file), 'rate_units': 1}
    death = ss.Deaths(death_rates)
    ppl = ss.People(population_size, age_data=demographics_data)

    # Create the networks - sexual and maternal
    mf = ss.MFNet(duration=1/24, acts=80)
    maternal = ss.MaternalNet()
    networks = [mf, maternal]

    # Initialize diseases
    hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=0.0001)
    t2d_disease = mi.Type2Diabetes(init_prev=ss.bernoulli(get_prevalence_function('Type2Diabetes')))
    ckd_disease = mi.ChronicKidneyDisease(init_prev=ss.bernoulli(get_prevalence_function('ChronicKidneyDisease')))

    # Read interaction data and create connectors dynamically
    rel_sus = mi.read_interactions('../mighti/data/rel_sus.csv')
    interactions = mi.create_connectors(rel_sus)

    # Initialize the simulation with connectors
    sim = ss.Sim(
        n_agents=population_size,
        networks=networks,
        diseases=[hiv_disease, t2d_disease, ckd_disease],
        analyzers=[prevalence_analyzer],
        start=init_year,
        stop=end_year,
        people=ppl,
        demographics=[pregnancy, death],
        connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )

    # Run the simulation
    sim.run()
    return sim, prevalence_analyzer

def plot_results(sim, prevalence_analyzer, outcome):
    if outcome == "Mean Prevalence":
        mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')
        mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ChronicKidneyDisease')
    elif outcome == "Population":
        st.write("Population outcome is not implemented yet.")
    elif outcome == "Sex-Dependent Prevalence":
        st.write("Sex-dependent prevalence outcome is not implemented yet.")