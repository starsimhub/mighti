import starsim as ss
import stisim as sti
import mighti as mi
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

n_agents = 10_000
inityear = 2000
endyear = 2007
country = "Eswatini"
disease = "Type2Diabetes"
diseases = ['HIV', disease]
csv_prevalence = "../mighti/data/prevalence_data_eswatini.csv"
csv_path_age = '../mighti/data/eswatini_age_distribution_2007.csv'
csv_path_params = '../mighti/data/eswatini_parameters_dt.csv'

def make_sim(incidence_prob):

    # Load the age distribution data for the specified year
    age_distribution_year = pd.read_csv(csv_path_age)

    # Load parameters
    df = pd.read_csv(csv_path_params)
    df.columns = df.columns.str.strip()
    
    prevalence_data_df = pd.read_csv(csv_prevalence)

    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases, prevalence_data=prevalence_data_df, inityear=inityear
    )

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(
            disease, prevalence_data, age_bins, sim, size)

    # Initialize the PrevalenceAnalyzer
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

    ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))
    
    # Initialize networks
    # mf = ss.MFNet(duration=1/24, acts=80)
    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]

    hiv = sti.HIV(init_prev=ss.bernoulli(get_prev_fn('HIV')), init_prev_data=None,
                  p_hiv_death=0, include_aids_deaths=False,
                  beta={'structuredsexual': [0.001, 0.001], 'maternal': [0.01, 0.01]})
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    disease_class = getattr(mi, disease, None)
    if disease_class:
        disease_obj = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev})
        disease_objects.append(disease_obj)
    disease_objects.append(hiv)

    sim = ss.Sim(
        dt=1/12,
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        analyzers=[prevalence_analyzer],
        diseases=disease_objects,
        people=ppl,
        copy_inputs=False,
        # verbose=0,
    )
    return sim

def extract_prevalence(sim):
    ppl = sim.people
    age_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40),
                (40, 45), (45, 50), (50, 55), (55, 60), (60, 65),
                (65, 70), (70, 75), (75, 80), (80, float('inf'))]
    records = []
    has_disease_f = t2d.affected & ppl.female
    has_disease_m = t2d.affected & ppl.male
    for age_start, age_end in age_bins:
        age_group = (ppl.age >= age_start) & (ppl.age < age_end)
        num_male = np.sum(age_group & has_disease_m)
        den_male = np.sum(age_group & ppl.male)
        num_female = np.sum(age_group & has_disease_f)
        den_female = np.sum(age_group & ppl.female)
        prev_male = num_male / den_male if den_male > 0 else 0.0
        prev_female = num_female / den_female if den_female > 0 else 0.0
        records.append({"Age": age_start, "Sex": "male", "prevalence": prev_male})
        records.append({"Age": age_start, "Sex": "female", "prevalence": prev_female})
    df = pd.DataFrame(records)
    df = df.set_index(["Age", "Sex"]).sort_index()
    return df

def get_observed_prevalence():
    prevalence_data_df = pd.read_csv(csv_prevalence)
    diseases = [disease]
    prevalence_dict, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data_df, inityear)
    expected = []
    for age in age_bins[disease]:
        for sex in ['male', 'female']:
            prev = prevalence_dict[disease][sex][age]
            expected.append({'Age': age, 'Sex': sex, 'prevalence': prev})
    observed_df = pd.DataFrame(expected)
    observed_df = observed_df.set_index(["Age", "Sex"]).sort_index()
    return observed_df

def loss_fn(inc_rate_value, verbose=False):
    sim, t2d = make_sim(inc_rate_value)
    sim.run()
    pred_df = extract_prevalence(sim, t2d)
    obs_df = get_observed_prevalence()
    # Align indices
    pred_df, obs_df = pred_df.align(obs_df, join='inner')
    diff = pred_df['prevalence'] - obs_df['prevalence']
    if verbose:
        print(f"incidence={inc_rate_value:.6f}, loss={np.sum(diff**2):.6f}")
    return np.sum(diff**2)

if __name__ == "__main__":
    # Use bounded scalar minimization (log-scale bounds if desired)
    result = minimize_scalar(loss_fn, bounds=(0.001, 0.05), method="bounded", options={'xatol': 1e-4})
    print("\nBest-fit incidence rate:", result.x)
    print("Best-fit loss:", result.fun)
    # Optionally, print the prevalence fit for the best value
    sim, t2d = make_sim(result.x)
    sim.run()
    pred_df = extract_prevalence(sim, t2d)
    obs_df = get_observed_prevalence()
    print("\nPredicted prevalence (best fit):\n", pred_df)
    print("\nObserved prevalence:\n", obs_df)