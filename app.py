import streamlit as st
import mighti as mi
import pandas as pd

# Sidebar for file uploads
st.sidebar.title("Upload Files")
prevalence_file = st.sidebar.file_uploader("Upload Age-Sex Dependent Prevalence Data", type="csv", key="prevalence_file_key")
mortality_file = st.sidebar.file_uploader("Upload Mortality Data", type="csv", key="mortality_file_key")
fertility_file = st.sidebar.file_uploader("Upload Fertility Data", type="csv", key="fertility_file_key")
demographics_file = st.sidebar.file_uploader("Upload Demographics Data", type="csv", key="demographics_file_key")

# Sidebar for simulation parameters
st.sidebar.title("Simulation Parameters")
init_year = st.sidebar.number_input("Initial Year", value=2007, min_value=1900, max_value=2100, key="init_year_key")
end_year = st.sidebar.number_input("End Year", value=2050, min_value=1900, max_value=2100, key="end_year_key")
population_size = st.sidebar.number_input("Population Size", value=5000, min_value=1, key="population_size_key")
area = st.sidebar.text_input("Area (Country, City, State)", value="Eswatini", key="area_key")

# Sidebar for outcomes
st.sidebar.title("Outcomes")
outcome = st.sidebar.selectbox("Select Outcome", ["Mean Prevalence", "Population", "Sex-Dependent Prevalence"], key="outcome_key")

# Main section
st.title("MIGHTI Simulation")

if st.sidebar.button("Run Simulation"):
    if prevalence_file and demographics_file and mortality_file and fertility_file:
        prevalence_data = pd.read_csv(prevalence_file)
        demographics_data = pd.read_csv(demographics_file)
        fertility_data = pd.read_csv(fertility_file)
        mortality_data = pd.read_csv(mortality_file)

        sim, prevalence_analyzer = mi.run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size)
        mi.plot_results(sim, prevalence_analyzer, outcome)
    else:
        st.error("Please upload all required files.")
        
        
        
# source streamlit_env/bin/activate

# # Install Streamlit and dependencies
# pip install --upgrade pip
# pip install streamlit
# pip install pyarrow

# # Run the Streamlit app
# streamlit run app.py           