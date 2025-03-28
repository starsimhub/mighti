import starsim as ss
import sciris as sc
import mighti as mi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
beta = 0.001
n_agents = 50000  # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2021

# ---------------------------------------------------------------------
# Specify data file paths
# ---------------------------------------------------------------------

# Parameters
csv_path_params =  'mighti/data/eswatini_parameters.csv'

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus_0.csv"

# Prevalence data
csv_prevalence = 'mighti/data/prevalence_data_eswatini.csv'

# Fertility data 
csv_path_fertility = 'mighti/data/eswatini_asfr.csv'

# Death data
csv_path_death = 'mighti/data/eswatini_deaths.csv'

# Age distribution data
csv_path_age = 'mighti/data/eswatini_age_2023.csv'

df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()



# Define diseases
# conditions = ncds + communicable_diseases  # List of all diseases

ncd = ['Type2Diabetes']#, 'ChronicKidneyDisease','CervicalCancer','ProstateCancer'] 
diseases = ['HIV'] + ncd #+conditions # List of diseases including HIV


# Load prevalence data from the CSV file
prevalence_data_df = pd.read_csv(csv_prevalence)

#Check that there are non-zero values for CCa, T2D, CKD
# only prostate cancer data is being plotted for observed scatter point plots
print(prevalence_data_df[['ProstateCancer_male', 'CervicalCancer_male', 'Type2Diabetes_male', 'ChronicKidneyDisease_male']].tail())

# Initialize prevalence data from the DataFrame
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data=prevalence_data_df, inityear=inityear)

print(f"Loaded prevalence data for Type2Diabetes: {prevalence_data['Type2Diabetes']}")

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(sc.thispath() / 'mighti/data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'mighti/data/eswatini_deaths.csv'), 'rate_units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('mighti/data/eswatini_age_2023.csv'))

# Create the networks - sexual and maternal
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# Add this at the top of your script where you're instantiating disease objects
print("=====================================================")
print("DEBUGGING TYPE2DIABETES INITIALIZATION PREVALENCE")
print("=====================================================")

# 1. First check the prevalence data loaded from CSV
print("\n1. CHECKING PREVALENCE DATA FROM CSV:")
print(f"Initial year: {inityear}")  # Should be 2007
t2d_data = prevalence_data_df[prevalence_data_df['Year'] == inityear]
print("Type2Diabetes data for init year:")
print(t2d_data[['Year', 'Type2Diabetes_male', 'Type2Diabetes_female']].head())

# 2. Check the processed prevalence data structure
print("\n2. CHECKING PROCESSED PREVALENCE DATA STRUCTURE:")
if 'Type2Diabetes' in prevalence_data:
    print("Male prevalence values by age:")
    for age, value in sorted(prevalence_data['Type2Diabetes']['male'].items()):
        print(f"  Age {age}: {value}")
    
    # Check a sample female prevalence
    print("\nFemale prevalence values (first few ages):")
    sample_ages = list(sorted(prevalence_data['Type2Diabetes']['female'].keys()))[:5]
    for age in sample_ages:
        print(f"  Age {age}: {prevalence_data['Type2Diabetes']['female'][age]}")
else:
    print("ERROR: Type2Diabetes not found in prevalence_data!")

# Testing the prevalence function directly
print("3. TESTING get_prevalence_function DIRECTLY:")
prevalence_func = get_prevalence_function('Type2Diabetes')

# Create a proper mock simulation for testing
class MockSim:
    def __init__(self):
        self.people = type('', (), {})()
        self.mock_data = {
            'age': np.array([20, 30, 40, 50, 60]),
            'female': np.array([True, False, True, False, True]),
        }
        self.people.age = self.mock_data['age']
        self.people.female = self.mock_data['female']

mock_sim = MockSim()

# Now test with the mock sim
try:
    print("Testing with mock simulation...")
    test_results = prevalence_func(None, mock_sim, np.arange(5))
    print(f"Test prevalence function results: {test_results}")
except Exception as e:
    print(f"Error testing prevalence function: {str(e)}")
    import traceback
    traceback.print_exc()

# 4. Check type of init_prev
print("\n4. CHECKING init_prev TYPE AND BEHAVIOR:")
init_prev_t2d = ss.bernoulli(get_prevalence_function('Type2Diabetes'))
print(f"Type of init_prev: {type(init_prev_t2d)}")
print(f"init_prev attributes: {dir(init_prev_t2d)}")


    

hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)

# # Automatically create disease objects for NCDs
# disease_objects = []
# for disease in ncd:
#     init_prev = ss.bernoulli(get_prevalence_function(disease))
    
#     # Dynamically get the disease class from `mi` module
#     disease_class = getattr(mi, disease, None)
    
#     if disease_class:
#         disease_obj = disease_class(init_prev=init_prev)  # Instantiate dynamically
#         disease_objects.append(disease_obj)
#     else:
#         print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")

# Automatically create disease objects for all diseases
disease_objects = []
for disease in ncd:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    
    # Dynamically get the disease class from `mi` module
    disease_class = getattr(mi, disease, None)
    
    if disease_class:
        disease_obj = disease_class(disease_name=disease, csv_path=csv_path_params, pars={"init_prev": init_prev})  # Instantiate dynamically and pass csv_path
        disease_objects.append(disease_obj)
    else:
        print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")


# Combine all disease objects including HIV
disease_objects.append(hiv_disease)
# 5. Add this right after creating the Type2Diabetes instance:
print("\n5. CHECKING DISEASE OBJECT PARAMETERS:")
# Find the Type2Diabetes instance in disease_objects
for disease_obj in disease_objects:
    if hasattr(disease_obj, 'disease_name') and disease_obj.disease_name == 'Type2Diabetes':
        print(f"Found Type2Diabetes object")
        # Print the parameters
        print(f"init_prev type: {type(disease_obj.pars.init_prev)}")
        print(f"init_prev from pars: {disease_obj.pars.init_prev}")
        
        # Check what parameters were loaded from CSV
        if hasattr(disease_obj, 'get_disease_parameters'):
            params = disease_obj.get_disease_parameters()
            print(f"Parameters from CSV: {params}")
            print(f"init_prev from CSV: {params.get('init_prev', 'Not found')}")
        else:
            print("No get_disease_parameters method")
        break
else:
    print("Type2Diabetes object not found in disease_objects!")


# Initialize interaction objects for HIV-NCD interactions
ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
interactions = [ncd_hiv_connector]

# Load NCD-NCD interactions
ncd_interactions = mi.read_interactions(csv_path_interactions)  # Reads rel_sus_0.csv
connectors = mi.create_connectors(ncd_interactions)

# Add NCD-NCD connectors to interactions
interactions.extend(connectors)
     
# Existing imports and code...

if __name__ == '__main__':
    # Initialize the simulation with connectors
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        diseases=disease_objects,
        analyzers=[prevalence_analyzer],
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        # connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )

    # Run the simulation
    sim.run()

    
    # 6. Add this after running the simulation:
    print("\n6. CHECKING FINAL PREVALENCE RESULTS:")
    if hasattr(prevalence_analyzer, 'results'):
        # Find the keys related to Type2Diabetes
        t2d_keys = [k for k in prevalence_analyzer.results.keys() if 'Type2Diabetes' in k]
        # print(f"Available T2D result keys: {t2d_keys}")
        
        # Print the first 3 values for prevalence metrics if they exist
        for key in ['Type2Diabetes_num_male_0', 'Type2Diabetes_den_male_0']:
            if key in prevalence_analyzer.results:
                print(f"{key} (first 3 timepoints): {prevalence_analyzer.results[key][:3]}")
        
        # Calculate the actual prevalence for the first timepoint
        for i in range(len(prevalence_analyzer.age_bins)):
            num_key = f'Type2Diabetes_num_male_{i}'
            den_key = f'Type2Diabetes_den_male_{i}'
            if num_key in prevalence_analyzer.results and den_key in prevalence_analyzer.results:
                num = prevalence_analyzer.results[num_key][0]
                den = prevalence_analyzer.results[den_key][0]
                total = num + den
                if total > 0:
                    prevalence = num / total *100
                    print(f"Age bin {i} initial prevalence: {prevalence:.2f}% (num={num}, den={den})")
    else:
        print("No results found in prevalence_analyzer!")
    print("Simulation complete. Plotting results...")
    
    # Print the simulated prevalence for the inityear
    def print_simulated_prevalence_for_year(sim, prevalence_analyzer, year):
        time_index = np.where(sim.timevec == year)[0]
        if len(time_index) == 0:
            print(f"Year {year} not found in simulation time vector.")
            return
        
        time_index = time_index[0] # Get the first occurrence
        print(f"Simulated prevalence for the year {year}:")

        for disease in diseases:
            male_key = f'{disease}_num_male_{time_index}'
            female_key = f'{disease}_num_female_{time_index}'
            male_den_key = f'{disease}_den_male_{time_index}'
            female_den_key = f'{disease}_den_female_{time_index}'

            male_prevalence = np.nan_to_num(prevalence_analyzer.results.get(male_key, 0) / prevalence_analyzer.results.get(male_den_key, 1))*100
            female_prevalence = np.nan_to_num(prevalence_analyzer.results.get(female_key, 0) / prevalence_analyzer.results.get(female_den_key, 1))*100

            print(f"{disease}: Male Prevalence: {male_prevalence}%, Female Prevalence: {female_prevalence}%")
    
    # Call the function to print the simulated prevalence for the inityear
    print_simulated_prevalence_for_year(sim, prevalence_analyzer, inityear)

    # Plot the results for each simulation
    mi.plot_mean_prevalence_afia(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, inityear, endyear)  
    # mi.plot_mean_prevalence_afia(sim, prevalence_analyzer, 'ChronicKidneyDisease', prevalence_data_df, inityear, endyear)
    # mi.plot_mean_prevalence_afia(sim, prevalence_analyzer, 'CervicalCancer', prevalence_data_df, inityear, endyear)
    # mi.plot_mean_prevalence_afia(sim, prevalence_analyzer, 'ProstateCancer', prevalence_data_df, inityear, endyear)
 
    mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')  

