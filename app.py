import streamlit as st
import pandas as pd
import mighti as mi
import prepare_data_for_year
from mighti.app_data_cleaning import data_extraction

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #57068c;
        color: #FFFFFF;
        font-weight: bold;
    }
    .custom-upload-title {
        font-size: 30px;
        color: #57068c; /* NYU violet */
        font-weight: bold;
        margin-bottom: 0px;
    }
    .custom-upload-box {
        margin-top: -30px; /* Adjust this value to increase space between upload box and next title */
        margin-bottom: 30px; /* Adjust this value to increase space between upload box and next title */
    }
    .custom-title {
        color: #57068c;
    }
    .sidebar .sidebar-content {
        width: 350px; /* Adjust the width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Run Life Expectancy Simulation", "Run Demo", "Run Disease Simulation", "Documentation"])

# Function to prepare data
def prepare_data(country, year):
    prepare_data_for_year.prepare_data_for_year(year)
    # Load the necessary data files based on the selected country and year
    csv_path_params = f'demography/{country}/parameters.csv'
    csv_path_death = f'demography/{country}/mortality_rates_{year}.csv'
    csv_path_age = f'demography/{country}/age_distribution_{year}.csv'
    csv_path_fertility = f'demography/{country}/_asfr.csv'
    csv_path_prevalence = f'demography/{country}/prevalence_data.csv'
    return csv_path_params, csv_path_death, csv_path_age, csv_path_fertility, csv_path_prevalence

def ensure_odd(value):
    return value if value % 2 != 0 else value + 1  # Make the number odd if it's even
    
if page == "Run Life Expectancy Simulation":
    # Sidebar for simulation parameters
    st.sidebar.title("Simulation Parameters")
    
    # Read the CSV data into a DataFrame
    df = pd.read_csv('demography/death_single_age_female.csv')
    
    # Extract unique regions
    unique_regions = df['region'].unique()
    
    # Create a sidebar selectbox with the unique regions
    country = st.sidebar.selectbox('Select a Region:', unique_regions)   
    st.sidebar.write("Please choose an odd number for Initial and End Year:")
    init_year = st.sidebar.number_input(
        "Initial Year", 
        value=ensure_odd(2007), 
        min_value=ensure_odd(1900), 
        max_value=ensure_odd(2023), 
        key="init_year_key")

    # Sidebar input for End Year
    end_year = st.sidebar.number_input(
        "End Year",
        value=ensure_odd(2021),  # Ensure the default value is odd
        min_value=ensure_odd(1901),  # Ensure minimum value is odd
        max_value=ensure_odd(2023),  # Ensure maximum value is odd
        key="end_year_key")
    
    # Sidebar input for Population Size
    population_size = st.sidebar.number_input(
        "Population Size",
        value=ensure_odd(5001),  # Ensure the default value is odd
        min_value=ensure_odd(1),  # Ensure minimum value is odd
        key="population_size_key")

    
    # Choose the outcome to plot
    st.sidebar.title("Outcomes")
    outcome = st.sidebar.selectbox("Select Outcome", ["Mortality Rates", "Life Expectancy"], key="outcome_key")


    # Main section
    st.markdown('<h1 class="custom-title">Life Expectancy Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<h2>Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)</h2>', unsafe_allow_html=True)
    st.info(
        """
        Please select the parameters in the sidebar. Once you have done so, click the "Run Simulation" button 
        to see the results here.
        """
    )
    
    # Placeholder for results
    results_placeholder = st.empty()
    
    if st.sidebar.button("Run Simulation"):
        # Use data_extraction to prepare the data
        (
            population_csv,
            fertility_csv,
            mortality_csv,
            mortality_rates_csv,
            life_expectancy_csv,
            extracted_life_expectancy_csv
        ) = data_extraction(country, init_year, end_year)
        
        # Run the simulation
        sim, df_metrics, life_table = mi.run_demography(
            population_csv, fertility_csv, mortality_rates_csv, life_expectancy_csv, extracted_life_expectancy_csv, country, init_year, end_year, population_size
        )
        mi.plot_demography(outcome, df_metrics, mortality_csv, life_table, extracted_life_expectancy_csv, end_year)
 
        
if page == "Run Demo":
    st.sidebar.title("Simulation Parameters")
    init_year = 2007  # Fixed initial year to 2007

    # Sidebar input for End Year
    end_year = st.sidebar.number_input(
        "End Year",
        value=ensure_odd(2021),  # Ensure the default value is odd
        min_value=ensure_odd(1901),  # Ensure minimum value is odd
        max_value=ensure_odd(2023),  # Ensure maximum value is odd
        key="end_year_key"
    )
    
    # Sidebar input for Population Size
    population_size = st.sidebar.number_input(
        "Population Size",
        value=ensure_odd(5001),  # Ensure the default value is odd
        min_value=ensure_odd(1),  # Ensure minimum value is odd
        key="population_size_key"
    )

    # Sidebar for outcomes
    st.sidebar.title("Outcomes")
    outcome = st.sidebar.selectbox("Select Outcome", ["Mean Prevalence", "Age-dependent Prevalence"], key="outcome_key")

    # Load the parameters data to get the list of diseases
    csv_path_params = 'mighti/data/eswatini_parameters.csv'
    df = pd.read_csv(csv_path_params)
    healthconditions = df['condition'].unique().tolist()
    
    disease = st.sidebar.selectbox("Select Disease to Plot", healthconditions, key="disease")

    # Main section
    st.markdown('<h1 class="custom-title">MIGHTI Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<h3>Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)</h2>', unsafe_allow_html=True)
    st.write(
        "This demo showcases disease prediction using Eswatini's initial prevalence data. "
        "If similar prevalence data is available for another region, the model can simulate outcomes for that region as well."
    )
    st.info(
        """
        Please set the simulation parameters in the sidebar. Once you have done so, click the "Run Demo" button 
        to see the results here.
        """
    )
    # Placeholder for results
    results_placeholder = st.empty()
    
    if st.sidebar.button("Run Demo"):
        # Load predefined data for the demo
        prevalence_data = pd.read_csv('mighti/data/prevalence_data_eswatini.csv')
        demographics_data = pd.read_csv('mighti/data/eswatini_age_distribution_2007.csv')
        fertility_data = pd.read_csv('mighti/data/eswatini_asfr.csv')
        mortality_data = pd.read_csv('mighti/data/eswatini_mortality_rates_2007.csv')
        age_bins = [(0, 5, "0-4"), (5, 15, "5-14"), (15, 25, "15-24"), (25, 35, "25-34"),
                    (35, 45, "35-44"), (45, 55, "45-54"), (55, 65, "55-64"), (65, 75, "65-74"),
                    (75, 85, "75-84"), (85, 101, "85+")]

        sim, prevalence_analyzer = mi.run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size)
        mi.plot_results(sim, prevalence_analyzer, outcome, disease, age_bins, prevalence_data, init_year, end_year)


 
elif page == "Run Disease Simulation":
    
    # Sidebar for file uploads
    st.sidebar.title("Upload Files")

    st.sidebar.markdown('<p class="custom-upload-title">Age-Sex Dependent Prevalence Data</p>', unsafe_allow_html=True)
    prevalence_file = st.sidebar.file_uploader("", type="csv", key="prevalence_file_key")
    st.sidebar.markdown('<div class="custom-upload-box"></div>', unsafe_allow_html=True)  # Add space after upload box

    st.sidebar.markdown('<p class="custom-upload-title">Mortality Data</p>', unsafe_allow_html=True)
    mortality_file = st.sidebar.file_uploader("", type="csv", key="mortality_file_key")
    st.sidebar.markdown('<div class="custom-upload-box"></div>', unsafe_allow_html=True)  # Add space after upload box

    st.sidebar.markdown('<p class="custom-upload-title">Fertility Data</p>', unsafe_allow_html=True)
    fertility_file = st.sidebar.file_uploader("", type="csv", key="fertility_file_key")
    st.sidebar.markdown('<div class="custom-upload-box"></div>', unsafe_allow_html=True)  # Add space after upload box

    st.sidebar.markdown('<p class="custom-upload-title">Demographics Data</p>', unsafe_allow_html=True)
    demographics_file = st.sidebar.file_uploader("", type="csv", key="demographics_file_key")
    st.sidebar.markdown('<div class="custom-upload-box"></div>', unsafe_allow_html=True)  # Add space after upload box

    # st.sidebar.markdown('<p class="custom-upload-title">Parameters </p>', unsafe_allow_html=True)
    # parameters_file = st.sidebar.file_uploader("", type="csv", key="parameters_file_key")
    # st.sidebar.markdown('<div class="custom-upload-box"></div>', unsafe_allow_html=True)  # Add space after upload box

    # Sidebar for simulation parameters
    st.sidebar.title("Simulation Parameters")
    init_year = 2007  # Fixed initial year to 2007
    end_year = st.sidebar.number_input("End Year", value=ensure_odd(2023), min_value=ensure_odd(1900), max_value=ensure_odd(2023), key="end_year_key")
    population_size = st.sidebar.number_input("Population Size", value=5000, min_value=1, key="population_size_key")
    
    # Sidebar for outcomes
    st.sidebar.title("Outcomes")
    outcome = st.sidebar.selectbox("Select Outcome", ["Mean Prevalence", "Age-dependent Prevalence", "Life Expectancy"], key="outcome_key")

    # Load the parameters data to get the list of diseases
    csv_path_params = 'mighti/data/eswatini_parameters.csv'
    df = pd.read_csv(csv_path_params)
    healthconditions = df['condition'].unique().tolist()
    
    disease = st.sidebar.selectbox("Select Disease to Plot", healthconditions, key="disease")

    # Main section
    st.markdown('<h1 class="custom-title">MIGHTI Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<h2>Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)</h2>', unsafe_allow_html=True)
    st.info(
        """
        Please set the simulation parameters in the sidebar. Once you have done so, click the "Run Simulation" button 
        to see the results here.
        """
    )
   
    # Placeholder for results
    results_placeholder = st.empty()
    
    if st.sidebar.button("Run Simulation"):
        # Load predefined data for the simulation
        prevalence_data = pd.read_csv('mighti/data/prevalence_data_eswatini.csv')
        demographics_data = pd.read_csv('mighti/data/eswatini_age_2023.csv')
        fertility_data = pd.read_csv('mighti/data/eswatini_asfr.csv')
        mortality_data = pd.read_csv('mighti/data/eswatini_deaths.csv')
        age_bins = [(0, 5, "0-4"), (5, 15, "5-14"), (15, 25, "15-24"), (25, 35, "25-34"),
                    (35, 45, "35-44"), (45, 55, "45-54"), (55, 65, "55-64"), (65, 75, "65-74"),
                    (75, 85, "75-84"), (85, 101, "85+")]

        sim, prevalence_analyzer = mi.run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size)
        mi.plot_results(sim, prevalence_analyzer, outcome, disease, age_bins)

elif page == "Documentation":
    st.title("Documentation")
    st.write("""
    ## Data Requirements
    The following CSV files are required to run the simulation:

    1. **Age-Sex Dependent Prevalence Data**: This file should contain the prevalence data stratified by age and sex.
    2. **Mortality Data**: This file should contain the mortality rates for different age groups.
    3. **Fertility Data**: This file should contain the fertility rates for different age groups.
    4. **Demographics Data**: This file should contain the demographic information for the population.
    5. **Parameters Data**: This file should contain the parameters for different diseases.
    6. **Relative Risks Data**: This file should contain the relative risks for interactions between diseases.

    ## File Format
    Each CSV file should have the following columns:

    ### Age-Sex Dependent Prevalence Data
    - `age_group`: Age group of the individuals.
    - `sex`: Sex of the individuals (e.g., Male, Female).
    - `prevalence`: Prevalence rate for the specific age group and sex.

    ### Mortality Data
    - `age_group`: Age group of the individuals.
    - `mortality_rate`: Mortality rate for the specific age group.

    ### Fertility Data
    - `age_group`: Age group of the individuals.
    - `fertility_rate`: Fertility rate for the specific age group.

    ### Demographics Data
    - `age_group`: Age group of the individuals.
    - `population`: Number of individuals in the specific age group.

    ### Parameters Data
    - `condition`: Name of the disease or condition.
    - `p_death`: Probability of death due to the condition.
    - `incidence`: Incidence rate of the condition.
    - `dur_condition`: Duration of the condition.
    - `init_prev`: Initial prevalence rate of the condition.
    - `rel_sus`: Relative susceptibility to the condition.
    - `remission_rate`: Remission rate of the condition.
    - `max_disease_duration`: Maximum duration of the condition.

    ### Relative Risks Data
    - `condition`: Name of the condition.
    - `rel_sus`: Relative susceptibility for interactions between diseases.
    """)        
# source streamlit_env/bin/activate

# # Install Streamlit and dependencies
# pip install --upgrade pip
# pip install streamlit
# pip install pyarrow

# # Run the Streamlit app
# streamlit run app.py          