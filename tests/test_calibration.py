import starsim as ss
import stisim as sti
import mighti as mi
import pandas as pd
import numpy as np
import sciris as sc
import pytest

n_agents = 10_000
inityear = 2007
endyear = 2010
country = "Eswatini"
disease = "Type2Diabetes"
diseases = ['HIV', disease]
csv_prevalence = "../mighti/data/prevalence_data_eswatini.csv"
csv_path_age = '../mighti/data/eswatini_age_distribution_2007.csv'
csv_path_params = '../mighti/data/eswatini_parameters_dt.csv'

def make_sim():
   
    # # Load parameters
    # df = pd.read_csv(csv_path_params)
    # df.columns = df.columns.str.strip()
    
    prevalence_data_df = pd.read_csv(csv_prevalence)

    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases, prevalence_data=prevalence_data_df, inityear=inityear
    )

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(
            disease, prevalence_data, age_bins, sim, size)

    ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))
    
    # Initialize networks
    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]

    disease_objects = []

    hiv = sti.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), init_prev_data=None,
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
        people=ppl,
        diseases=disease_objects,
        copy_inputs=False,
        verbose=0,
    )
    return sim

def build_sim(sim, calib_pars=None, **kwargs):
    reps = kwargs.get('n_reps', 1)
    # Pick correct disease; adjust index as needed
    dis = sim.pars.diseases[1]  # or [0] if T2D is first

    # Set calibration parameters
    if calib_pars is not None:
        for k, v in calib_pars.items():
            if k == 'p_acquire':
                dis.pars['p_acquire'] = v
            else:
                raise NotImplementedError(f'Parameter {k} not recognized')

    # DO NOT run the sim here!
    if reps == 1:
        return sim

    ms = ss.MultiSim(sim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, reps)),
                     initialize=True, debug=True, parallel=False)
    return ms


@pytest.mark.calibration
def test_onepar_normal(do_plot=True):
    sc.heading('Testing a single parameter (beta) with a normally distributed likelihood')

    # Define the calibration parameters
    calib_pars = dict(
        p_acquire = dict(low=0.00001, high=0.1, guess=0.001, suggest_type='suggest_float', log=True),
    )

    sim = make_sim() 
    
    prevalence_data_df = pd.read_csv(csv_prevalence)
    diseases = [disease]
    prevalence_dict, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data_df, inityear)
    
    expected = []
    for age in age_bins[disease]:
        for sex in ['male', 'female']:
            prev = prevalence_dict[disease][sex][age]
            expected.append({'Age': age, 'Sex': sex, 'Year': inityear, 'prevalence': prev})
    
    expected_df = pd.DataFrame(expected)
    expected_df = expected_df.set_index(["Age", "Sex", "Year"]).sort_index()
    expected_agg = expected_df.reset_index().groupby('Year')['prevalence'].mean().to_frame()
    expected_agg.index.name = 't'
    expected_agg.index = expected_agg.index.astype(float)
    expected_agg['prevalence'] = expected_agg['prevalence'].astype(float)
    
    # This is the KEY step:
    expected_agg = expected_agg.rename(columns={'prevalence': 'x'})  
    
    def my_extract_fn(sim):
        vals = np.asarray(sim.results['type2diabetes']['new_cases'], dtype=float)
        tvec = np.asarray(sim.results['timevec'], dtype=float)
        df = pd.DataFrame({'x': vals}, index=pd.Index(tvec, name='t'))
        # Force column to float explicitly
        df['x'] = df['x'].astype(float)
        df.index = df.index.astype(float)
        print("extract_fn df.dtypes:", df.dtypes)
        print("extract_fn df.index.dtype:", df.index.dtype)
        return df
    
    prevalence_component = ss.Normal(
        name='NCD prevalence',
        conform='prevalent',
        expected=expected_agg,
        extract_fn=my_extract_fn,
    )

    calib = ss.Calibration(
        calib_pars=calib_pars,
        sim=sim,
        build_fn=build_sim,
        build_kw=dict(),
        reseed = False,
        components=[prevalence_component],
        total_trials=20,
        n_workers=None,
        debug=False,
        die=False,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    assert calib.check_fit(do_plot=False), 'Calibration did not improve the fit'

    # Call plotting to look for exceptions
    if do_plot:
        calib.plot_final()
        calib.plot(bootstrap=False)
        calib.plot(bootstrap=True)
        calib.plot_optuna(['plot_param_importances', 'plot_optimization_history'])

    return sim, calib

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    do_plot = True
    for f in [test_onepar_normal]:
        T = sc.timer()
        sim, calib = f(do_plot=do_plot)
        T.toc()
    plt.show()
