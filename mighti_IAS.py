import starsim as ss
import mighti as mi  
import pandas as pd
import pylab as pl
from copy import deepcopy

# Define diseases
ncds = ['Type2Diabetes']
diseases = ['HIV'] + ncds
beta = 0.001
n_agents = 50000
inityear = 2021
endyear = 2030

# -------------------------
# Prevalence Data
# -------------------------
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear
)

# -------------------------
# Demographics
# -------------------------
fertility_rates = {'fertility_rate': pd.read_csv('tests/test_data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv('tests/test_data/eswatini_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))

# -------------------------
# Networks
# -------------------------
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# -------------------------
# Disease Objects
# -------------------------
def get_prevalence_function(disease):
    """Get prevalence function for Type2Diabetes and HIV."""
    if disease == "Type2Diabetes":
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(
            "Type2Diabetes", prevalence_data, age_bins, sim, size
        )
    elif disease == "HIV":
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(
            "HIV", prevalence_data, age_bins, sim, size
        )
    else:
        raise ValueError(f"Prevalence function not defined for disease: {disease}")

# Create disease objects
init_prev_type2diabetes = ss.bernoulli(get_prevalence_function("Type2Diabetes"))
type2diabetes = mi.Type2Diabetes(init_prev=init_prev_type2diabetes)

init_prev_hiv = ss.bernoulli(get_prevalence_function("HIV"))
hiv = ss.HIV(init_prev=init_prev_hiv, beta=beta)

# List of disease objects
disease_objects = [type2diabetes, hiv]

# -------------------------
# Prevalence Analyzer
# -------------------------
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=["Type2Diabetes", "HIV"])

# -------------------------
# Simulation Without Interactions
# -------------------------
sim_no_interactions = ss.Sim(
    n_agents=n_agents,
    networks=deepcopy(networks),
    diseases=deepcopy(disease_objects),
    analyzers=[prevalence_analyzer],
    start=inityear,
    end=endyear,
    people=deepcopy(ppl),
    demographics=[pregnancy, death],
    copy_inputs=False
)

print("Running Simulation Without Interactions...")
sim_no_interactions.run()

# Extract Results
mortality_no_interactions = sim_no_interactions.results['new_deaths']
prevalence_results = prevalence_analyzer.results

# -------------------------
# Debugging Prevalence
# -------------------------
def extract_prevalence_by_gender(prevalence_analyzer, disease_name):
    """Extract prevalence for a specific disease by gender."""
    try:
        prevalence_male = prevalence_analyzer.results[f'{disease_name}_prevalence_male'] * 100
        prevalence_female = prevalence_analyzer.results[f'{disease_name}_prevalence_female'] * 100
        return prevalence_male, prevalence_female
    except KeyError as e:
        print(f"Prevalence data not found for {disease_name}: {e}")
        return None, None

# Plot Results
diseases_to_plot = ['HIV', 'Type2Diabetes']
fig, axs = pl.subplots(len(diseases_to_plot), 2, figsize=(15, len(diseases_to_plot) * 6))

for i, disease_name in enumerate(diseases_to_plot):
    prevalence_male, prevalence_female = extract_prevalence_by_gender(prevalence_analyzer, disease_name)

    # Plot Prevalence
    if prevalence_male is not None and prevalence_female is not None:
        axs[i, 0].plot(sim_no_interactions.yearvec, prevalence_male.mean(axis=1), label='Male', linestyle='-')
        axs[i, 0].plot(sim_no_interactions.yearvec, prevalence_female.mean(axis=1), label='Female', linestyle='--')
        axs[i, 0].set_title(f'Prevalence of {disease_name}', fontsize=16)
        axs[i, 0].set_xlabel('Year', fontsize=14)
        axs[i, 0].set_ylabel('Prevalence (%)', fontsize=14)
        axs[i, 0].legend()
        axs[i, 0].grid(True)

    # Plot Mortality
    axs[i, 1].plot(sim_no_interactions.yearvec, mortality_no_interactions, label=f'{disease_name} Mortality')
    axs[i, 1].set_title(f'Mortality of {disease_name}', fontsize=16)
    axs[i, 1].set_xlabel('Year', fontsize=14)
    axs[i, 1].set_ylabel('Mortality Count', fontsize=14)
    axs[i, 1].grid(True)

pl.tight_layout()
pl.show()
        
# import starsim as ss
# import mighti as mi  
# import pandas as pd
# import pylab as pl
# from copy import deepcopy

# # Define diseases
# ncds = ['Type2Diabetes']#, 'Obesity']
# diseases = ['HIV'] + ncds
# beta = 0.001
# n_agents = 50000
# inityear = 2021
# endyear = 2030

# # -------------------------
# # Prevalence Data
# # -------------------------
# prevalence_data, age_bins = mi.initialize_prevalence_data(
#     diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear
# )

# # -------------------------
# # Demographics
# # -------------------------
# fertility_rates = {'fertility_rate': pd.read_csv('tests/test_data/eswatini_asfr.csv')}
# pregnancy = ss.Pregnancy(pars=fertility_rates)
# death_rates = {'death_rate': pd.read_csv('tests/test_data/eswatini_deaths.csv'), 'units': 1}
# death = ss.Deaths(death_rates)
# ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))

# # -------------------------
# # Networks
# # -------------------------
# mf = ss.MFNet(duration=1/24, acts=80)
# maternal = ss.MaternalNet()
# networks = [mf, maternal]


# # -------------------------
# # Disease Objects
# # -------------------------

# def get_prevalence_function(disease):
#     """Get prevalence function for Type2Diabetes, Obesity, and HIV."""
#     if disease == "Type2Diabetes":
#         return lambda module, sim, size: mi.age_sex_dependent_prevalence(
#             "Type2Diabetes", prevalence_data, age_bins, sim, size
#         )
#     # elif disease == "Obesity":
#     #     return lambda module, sim, size: mi.age_sex_dependent_prevalence(
#     #         "Obesity", prevalence_data, age_bins, sim, size
#     #     )
#     elif disease == "HIV":
#         return lambda module, sim, size: mi.age_sex_dependent_prevalence(
#             "HIV", prevalence_data, age_bins, sim, size
#         )
#     else:
#         raise ValueError(f"Prevalence function not defined for disease: {disease}")

# # Create disease objects
# init_prev_type2diabetes = ss.bernoulli(get_prevalence_function("Type2Diabetes"))
# type2diabetes = mi.Type2Diabetes(init_prev=init_prev_type2diabetes)

# # init_prev_obesity = ss.bernoulli(get_prevalence_function("Obesity"))
# # obesity = mi.Obesity(init_prev=init_prev_obesity)

# init_prev_hiv = ss.bernoulli(get_prevalence_function("HIV"))
# hiv = ss.HIV(init_prev=init_prev_hiv, beta=beta)

# # List of disease objects
# disease_objects = [type2diabetes]

# # HIV Object
# hiv_init_prev = ss.bernoulli(get_prevalence_function("HIV"))
# hiv = ss.HIV(init_prev=hiv_init_prev, beta=beta)

# disease_objects.append(hiv)

# # -------------------------
# # Prevalence Analyzer
# # -------------------------

# prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=["Type2Diabetes", "HIV"])

# # -------------------------
# # Interactions
# # -------------------------

# # # Define interaction objects explicitly
# # interactions = [
# #     mi.hiv_type2diabetes(pars={"rel_sus_hiv_type2diabetes": 10}),
# #     # Uncomment if needed:
# #     # mi.hiv_obesity(pars={"rel_sus_hiv_obesity": 10}),
# # ]
# # # -------------------------
# # # Simulation With Interactions
# # # -------------------------
# # ppl_with_interactions = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))
# # networks_with_interactions = [deepcopy(net) for net in networks]
# # pregnancy_with_interactions = ss.Pregnancy(pars=fertility_rates)
# # death_with_interactions = ss.Deaths(death_rates)

# # disease_objects_with_interactions = []
# # for disease in ncds:
# #     disease_class = getattr(mi, disease)
# #     init_prev = ss.bernoulli(get_prevalence_function(disease))
# #     disease_obj = disease_class(init_prev=init_prev)
# #     disease_objects_with_interactions.append(disease_obj)

# # hiv_disease_with_interactions = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
# # disease_objects_with_interactions.append(hiv_disease_with_interactions)

# # sim_with_interactions = ss.Sim(
# #     n_agents=n_agents,
# #     networks=networks_with_interactions,
# #     diseases=disease_objects_with_interactions,
# #     analyzers=[prevalence_analyzer],
# #     start=inityear,
# #     end=endyear,
# #     connectors=interactions,
# #     people=ppl_with_interactions,
# #     demographics=[pregnancy_with_interactions, death_with_interactions],
# #     copy_inputs=False
# # )


# # print("Running Simulation With Interactions...")
# # sim_with_interactions.run()

# # mortality_with_interactions = sim_with_interactions.results['new_deaths']  # New deaths per time step
# # cumulative_mortality_with_interactions = sim_with_interactions.results['cum_deaths']  # Cumulative deaths

# # -------------------------
# # Simulation Without Interactions
# # -------------------------
# ppl_no_interactions = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))
# networks_no_interactions = [deepcopy(net) for net in networks]
# pregnancy_no_interactions = ss.Pregnancy(pars=fertility_rates)
# death_no_interactions = ss.Deaths(death_rates)

# disease_objects_no_interactions = deepcopy(disease_objects)

# sim_no_interactions = ss.Sim(
#     n_agents=n_agents,
#     networks=networks_no_interactions,
#     diseases=disease_objects_no_interactions,
#     analyzers=[prevalence_analyzer],
#     start=inityear,
#     end=endyear,
#     people=ppl_no_interactions,
#     demographics=[pregnancy_no_interactions, death_no_interactions],
#     copy_inputs=False
# )

# print("Running Simulation Without Interactions...")
# sim_no_interactions.run()

# mortality_no_interactions = sim_no_interactions.results['new_deaths']  # New deaths per time step
# cumulative_mortality_no_interactions = sim_no_interactions.results['cum_deaths']  # Cumulative deaths




# # # -------------------------
# # # Simulation With Interactions
# # # -------------------------
# # ppl_with_interactions = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))
# # networks_with_interactions = [deepcopy(net) for net in networks]
# # pregnancy_with_interactions = ss.Pregnancy(pars=fertility_rates)
# # death_with_interactions = ss.Deaths(death_rates)

# # disease_objects_with_interactions = deepcopy(disease_objects)

# # sim_with_interactions = ss.Sim(
# #     n_agents=n_agents,
# #     networks=networks_with_interactions,
# #     diseases=disease_objects_with_interactions,
# #     analyzers=[prevalence_analyzer],
# #     start=inityear,
# #     end=endyear,
# #     connectors=interactions,  # Include interactions
# #     people=ppl_with_interactions,
# #     demographics=[pregnancy_with_interactions, death_with_interactions],
# #     copy_inputs=False
# # )

# # print("Running Simulation With Interactions...")
# # sim_with_interactions.run()





# # -------------------------
# # Extract Mortality Data by Disease
# # -------------------------
# def extract_mortality(sim, disease_name):
#     """Extract mortality for a specific disease."""
#     disease_results = sim.results[disease_name.lower()]  # Disease-specific results
#     mortality = disease_results['new_deaths']  # Adjust the key if necessary
#     return mortality



# # -------------------------
# # Extract Prevalence Data by Gender
# # -------------------------
# def extract_prevalence_by_gender(prevalence_analyzer, disease_name):
#     """Extract prevalence for a specific disease by gender."""
#     try:
#         prevalence_male = prevalence_analyzer.results[f'{disease_name}_prevalence_male'] * 100
#         prevalence_female = prevalence_analyzer.results[f'{disease_name}_prevalence_female'] * 100
#         return prevalence_male, prevalence_female
#     except KeyError as e:
#         print(f"Prevalence data not found for {disease_name}: {e}")
#         return None, None

# # -------------------------
# # Plotting Mortality and Prevalence
# # -------------------------
# diseases_to_plot = ['HIV', 'Type2Diabetes']#, 'Obesity']
# fig, axs = pl.subplots(len(diseases_to_plot), 2, figsize=(15, len(diseases_to_plot) * 6))

# for i, disease_name in enumerate(diseases_to_plot):
#     # Mortality for the disease
#     mortality_no_interactions = sim_no_interactions.results['new_deaths']  # Adjust key if necessary
#     # mortality_with_interactions = sim_with_interactions.results['new_deaths']  # Adjust key if necessary
    
#     # Prevalence for the disease
#     prevalence_no_interactions_male, prevalence_no_interactions_female = extract_prevalence_by_gender(prevalence_analyzer, disease_name)
#     prevalence_with_interactions_male, prevalence_with_interactions_female = extract_prevalence_by_gender(prevalence_analyzer, disease_name)

#     # Plot mortality
#     axs[i, 0].plot(sim_no_interactions.yearvec, mortality_no_interactions, label='Without Interactions', linestyle='--')
#     # axs[i, 0].plot(sim_with_interactions.yearvec, mortality_with_interactions, label='With Interactions', linestyle='-')
#     axs[i, 0].set_title(f'Mortality Comparison: {disease_name}', fontsize=16)
#     axs[i, 0].set_xlabel('Year', fontsize=14)
#     axs[i, 0].set_ylabel('Mortality Rate (%)', fontsize=14)
#     axs[i, 0].legend()
#     axs[i, 0].grid(True)

#     # Plot prevalence
#     if prevalence_no_interactions_male is not None and prevalence_with_interactions_male is not None:
#         axs[i, 1].plot(sim_no_interactions.yearvec, prevalence_no_interactions_male.mean(axis=1), label='Male (Without Interactions)', linestyle='--')
#         # axs[i, 1].plot(sim_with_interactions.yearvec, prevalence_with_interactions_male.mean(axis=1), label='Male (With Interactions)', linestyle='-')
#         axs[i, 1].plot(sim_no_interactions.yearvec, prevalence_no_interactions_female.mean(axis=1), label='Female (Without Interactions)', linestyle='--')
#         # axs[i, 1].plot(sim_with_interactions.yearvec, prevalence_with_interactions_female.mean(axis=1), label='Female (With Interactions)', linestyle='-')
#         axs[i, 1].set_title(f'Prevalence Comparison: {disease_name}', fontsize=16)
#         axs[i, 1].set_xlabel('Year', fontsize=14)
#         axs[i, 1].set_ylabel('Prevalence (%)', fontsize=14)
#         axs[i, 1].legend()
#         axs[i, 1].grid(True)

# pl.tight_layout()
# pl.show()