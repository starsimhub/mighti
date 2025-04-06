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
st.markdown('<h1 class="custom-title">MIGHTI Simulation</h1>', unsafe_allow_html=True)
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
    if prevalence_file and demographics_file and mortality_file and fertility_file:
        prevalence_data = pd.read_csv(prevalence_file)
        demographics_data = pd.read_csv(demographics_file)
        fertility_data = pd.read_csv(fertility_file)
        mortality_data = pd.read_csv(mortality_file)

        sim, prevalence_analyzer = mi.run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size)
        mi.plot_results(sim, prevalence_analyzer, outcome)
    else:
        st.error("Please upload all required files.")