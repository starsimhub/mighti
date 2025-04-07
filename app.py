import streamlit as st
import pandas as pd
import mighti as mi

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
page = st.sidebar.radio("Go to", ["Run Simulation", "Run Demo", "Documentation"])

if page == "Run Simulation":
    
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
    end_year = st.sidebar.number_input("End Year", value=2030, min_value=1900, max_value=2100, key="end_year_key")
    population_size = st.sidebar.number_input("Population Size", value=5000, min_value=1, key="population_size_key")
    area = st.sidebar.text_input("Area (Country, City, State)", value="Eswatini", key="area_key")
    
    # Sidebar for outcomes
    st.sidebar.title("Outcomes")
    outcome = st.sidebar.selectbox("Select Outcome", ["Mean Prevalence", "Age-Dependent Prevalence", "Life Expectancy"], key="outcome_key")

    # Load the parameters data to get the list of diseases
    csv_path_params = 'mighti/data/eswatini_parameters.csv'
    df = pd.read_csv(csv_path_params)
    healthconditions = df['condition'].unique().tolist()
    
    disease = st.sidebar.selectbox("Select Disease to Plot", healthconditions, key="disease")

    # Main section
    st.markdown('<h1 class="custom-title">MIGHTI Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<h2>Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)</h2>', unsafe_allow_html=True)
    st.write(
        """
        Please upload the required data files and set the simulation parameters 
        in the sidebar. Once you have done so, click the "Run Simulation" button 
        to see the results here.
        """
    )
    
    # Placeholder for results
    results_placeholder = st.empty()
    
    if st.sidebar.button("Run Simulation"):
        if prevalence_file and demographics_file and mortality_file and fertility_file:# and parameters_file:
            prevalence_data = pd.read_csv(prevalence_file)
            demographics_data = pd.read_csv(demographics_file)
            fertility_data = pd.read_csv(fertility_file)
            mortality_data = pd.read_csv(mortality_file)
            # parameter_data = pd.read_csv(parameters_file)
            age_bins = [(0, 5, "0-4"), (5, 15, "5-14"), (15, 25, "15-24"), (25, 35, "25-34"),
                        (35, 45, "35-44"), (45, 55, "45-54"), (55, 65, "55-64"), (65, 75, "65-74"),
                        (75, 85, "75-84"), (85, 101, "85+")]

            sim, prevalence_analyzer = mi.run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size)
            mi.plot_results(sim, prevalence_analyzer, outcome, disease, age_bins)
        else:
            st.error("Please upload all required files.")
            
elif page == "Run Demo":
    # Sidebar for simulation parameters
    st.sidebar.title("Simulation Parameters")
    init_year = 2007  # Fixed initial year to 2007
    end_year = st.sidebar.number_input("End Year", value=2030, min_value=1900, max_value=2100, key="end_year_key")
    population_size = st.sidebar.number_input("Population Size", value=5000, min_value=1, key="population_size_key")
    #area = st.sidebar.text_input("Area (Country, City, State)", value="Eswatini", key="area_key")
    
    # Sidebar for outcomes
    st.sidebar.title("Outcomes")
    outcome = st.sidebar.selectbox("Select Outcome", ["Mean Prevalence", "Age-Dependent Prevalence", "Life Expectancy"], key="outcome_key")

    # Load the parameters data to get the list of diseases
    csv_path_params = 'mighti/data/eswatini_parameters.csv'
    df = pd.read_csv(csv_path_params)
    healthconditions = df['condition'].unique().tolist()
    
    disease = st.sidebar.selectbox("Select Disease to Plot", healthconditions, key="disease")

    # Main section
    st.markdown('<h1 class="custom-title">MIGHTI Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<h2>Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)</h2>', unsafe_allow_html=True)
    st.write(
        """
        Please upload the required data files and set the simulation parameters 
        in the sidebar. Once you have done so, click the "Run Simulation" button 
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