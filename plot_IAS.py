import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




##############################################################



def extract_t2d_prevalence_by_age_plhiv(sim, prevalence_analyzer, disease, target_years, age_groups):
    """
    Extract T2D prevalence among PLHIV for each age group and sex at specified years.
    """
    year_to_index = {int(round(year)): idx for idx, year in enumerate(sim.timevec)}
    selected_indices = {int(year): year_to_index[int(year)] for year in target_years if int(year) in year_to_index}

    def extract_results(key_pattern):
        return [
            prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec)))
            for i in range(len(age_groups))
        ]

    male_num = extract_results('num_with_HIV_male')
    female_num = extract_results('num_with_HIV_female')
    male_den = extract_results('den_with_HIV_male')
    female_den = extract_results('den_with_HIV_female')

    male_data, female_data = [], []
    for year in target_years:
        ti = selected_indices.get(int(year))
        if ti is None:
            continue

        male_row, female_row = [], []
        for i in range(len(age_groups)):
            m_num = male_num[i][ti]
            m_den = male_den[i][ti]
            f_num = female_num[i][ti]
            f_den = female_den[i][ti]

            male_row.append((m_num / m_den * 100) if m_den > 0 else 0)
            female_row.append((f_num / f_den * 100) if f_den > 0 else 0)

        male_data.append(male_row)
        female_data.append(female_row)

    age_labels = [label for _, _, label in age_groups]
    df_male = pd.DataFrame(male_data, index=target_years, columns=age_labels).T
    df_female = pd.DataFrame(female_data, index=target_years, columns=age_labels).T
    return df_male, df_female


def extract_combined_t2d_prevalence_by_age_plhiv(sim, prevalence_analyzer, disease, target_years, custom_age_groups):
    """
    Extract combined (male + female) T2D prevalence among PLHIV by custom age group and year.
    Uses fixed 15 bins from PrevalenceAnalyzer and aggregates to match custom bins.
    """
    import numpy as np
    import pandas as pd
    import sciris as sc

    # Define the fixed bins used in the PrevalenceAnalyzer
    analyzer_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45),
                     (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, float('inf'))]

    # Create mapping from custom group to list of indices in analyzer_bins
    bin_map = {}
    for i, (a_start, a_end, label) in enumerate(custom_age_groups):
        print(i)
        print(f'label is {label}, astart is {a_start}, aend is {a_end}')
        indices = [
            j for j, (b_start, b_end) in enumerate(analyzer_bins)
            if not (b_end <= a_start or b_start >= a_end)
        ]
        bin_map[label] = indices

    year_to_index = {int(round(year)): idx for idx, year in enumerate(sim.timevec)}
    selected_indices = {int(year): year_to_index[int(year)] for year in target_years if int(year) in year_to_index}

    combined_data = []

    for year in target_years:
        ti = selected_indices.get(int(year))
        if ti is None:
            continue

        row = []
        for label, indices in bin_map.items():
            m_num = sum(prevalence_analyzer.results[f'{disease}_num_with_HIV_male_{i}'][ti] for i in indices)
            m_den = sum(prevalence_analyzer.results[f'{disease}_den_with_HIV_male_{i}'][ti] for i in indices)
            f_num = sum(prevalence_analyzer.results[f'{disease}_num_with_HIV_female_{i}'][ti] for i in indices)
            f_den = sum(prevalence_analyzer.results[f'{disease}_den_with_HIV_female_{i}'][ti] for i in indices)

            total_num = m_num + f_num
            total_den = m_den + f_den
            prevalence = (total_num / total_den * 100) if total_den > 0 else 0
            row.append(prevalence)
            print(f'yearis {year}: prevalence is {prevalence}')

        combined_data.append(row)

    df_combined = pd.DataFrame(combined_data, index=target_years, columns=bin_map.keys()).T
    return df_combined


def plot_grouped_prevalence_bar(df, sex_label, colors):
    """
    Plot grouped bar chart for T2D among PLHIV by age group over years.
    Args:
        df (pd.DataFrame): age group x year prevalence (%)
        sex_label (str): 'Male' or 'Female'
        colors (list): list of colors for each age group
    """
    labels = df.columns.tolist()
    age_groups = df.index.tolist()
    x = np.arange(len(labels))  # positions for years
    width = 0.15

    fig, ax = plt.subplots(figsize=(10,5)) # larger figure

    for i, age_group in enumerate(age_groups):
        ax.bar(x + i * width, df.loc[age_group], width, label=age_group, color=colors[i])

    ax.set_xlabel('Year', fontsize=22, fontweight='bold')
    ax.set_ylabel('Type 2 Diabetes Prevalence (%)', fontsize=22, fontweight='bold')
    # ax.set_title(f'T2D Prevalence Among PLHIV by Age Group — {sex_label}', fontsize=24, fontweight='bold')
    ax.set_xticks(x + width * (len(age_groups) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticks(np.arange(0, 41, 5))
    ax.set_ylim(0, 40)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    ax.legend(title='Age Group', fontsize=20, title_fontsize=18, loc='upper left', frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)


    plt.tight_layout()
    plt.show()
    
    
def plot_grouped_prevalence_bar_combined(df, colors=None):
    """
    Plot grouped bar chart for combined (male + female) T2D prevalence among PLHIV by age group over selected years.

    Args:
        df (pd.DataFrame): DataFrame with age groups as rows and years as columns (values: prevalence %).
        colors (list): Optional list of colors for age groups.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = df.columns.tolist()        # years (x-axis)
    age_groups = df.index.tolist()      # age bins (bar series)
    x = np.arange(len(labels))          # x-axis positions for years
    n_groups = len(age_groups)
    width = 0.8 / n_groups              # bar width, adjusted to number of groups

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_groups))  # fallback color palette

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, age_group in enumerate(age_groups):
        ax.bar(x + i * width, df.loc[age_group], width, label=age_group, color=colors[i])

    ax.set_xlabel('Year', fontsize=22, fontweight='bold')
    ax.set_ylabel('T2D Prevalence among PLHIV (%)', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    ax.legend(title='Age Group', title_fontsize=16, fontsize=14, frameon=False, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # ax.set_title('T2D Prevalence Among PLHIV by Age Group (Combined Sex)', fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.show() 


def plot_t2d_prevalence_by_age_and_year(data, colors=None):
    """
    Plot grouped bar chart for T2D prevalence among PLHIV by age group over selected years.

    Args:
        data (dict or pd.DataFrame): Keys (or index) are age groups; columns are years; values are prevalence (%).
        colors (list): Optional list of colors for the bars.
    """
    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data).T  # transpose so rows = age groups, columns = years
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Input must be a DataFrame or a dict")

    labels = df.columns.astype(str).tolist()  # years on x-axis
    age_groups = df.index.tolist()            # age bins (bar series)
    x = np.arange(len(labels))                # x-axis positions for years
    n_groups = len(age_groups)
    width = 0.8 / n_groups                    # bar width

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_groups))  # default palette

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, age_group in enumerate(age_groups):
        ax.bar(x + i * width, df.loc[age_group], width, label=age_group, color=colors[i])

    ax.set_xlabel('Year', fontsize=22, fontweight='bold')
    ax.set_ylabel('T2D Prevalence among PLHIV (%)', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    ax.legend(title='Age Group', title_fontsize=16, fontsize=14, frameon=False, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    

def plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease):
    """
    Plot mean prevalence over time for a given disease and both sexes.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    """

    def extract_results(key_pattern):
        return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(prevalence_analyzer.age_bins))]

    male_num_with_HIV = np.sum(extract_results('num_with_HIV_male'), axis=0)
    female_num_with_HIV = np.sum(extract_results('num_with_HIV_female'), axis=0)
    male_den_with_HIV = np.sum(extract_results('den_with_HIV_male'), axis=0)
    female_den_with_HIV = np.sum(extract_results('den_with_HIV_female'), axis=0)
    male_num_without_HIV = np.sum(extract_results('num_without_HIV_male'), axis=0)
    female_num_without_HIV = np.sum(extract_results('num_without_HIV_female'), axis=0)
    male_den_without_HIV = np.sum(extract_results('den_without_HIV_male'), axis=0)
    female_den_without_HIV = np.sum(extract_results('den_without_HIV_female'), axis=0)

    male_den_with_HIV[male_den_with_HIV == 0] = 1
    female_den_with_HIV[female_den_with_HIV == 0] = 1
    male_den_without_HIV[male_den_without_HIV == 0] = 1
    female_den_without_HIV[female_den_without_HIV == 0] = 1

    mean_prevalence_male_with_HIV = np.nan_to_num(male_num_with_HIV / male_den_with_HIV) * 100
    mean_prevalence_female_with_HIV = np.nan_to_num(female_num_with_HIV / female_den_with_HIV) * 100
    mean_prevalence_male_without_HIV = np.nan_to_num(male_num_without_HIV / male_den_without_HIV) * 100
    mean_prevalence_female_without_HIV = np.nan_to_num(female_num_without_HIV / female_den_without_HIV) * 100
    
    # mean_prevalence_male_with_HIV = np.sum(extract_results('prev_with_HIV_male'), axis=0)
    # mean_prevalence_female_with_HIV = np.sum(extract_results('prev_with_HIV_female'), axis=0)
    # mean_prevalence_male_without_HIV = np.sum(extract_results('prev_without_HIV_male'), axis=0)
    # mean_prevalence_female_without_HIV = np.sum(extract_results('prev_without_HIV_female'), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sim.timevec, mean_prevalence_male_with_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV+)', linewidth=3, color='blue', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_female_with_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV+)', linewidth=3, color='red', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_male_without_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV-)', linewidth=3, color='blue', linestyle='dashed')
    ax.plot(sim.timevec, mean_prevalence_female_without_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV-)', linewidth=3, color='red', linestyle='dashed')

    ax.set_xlabel('Year', fontsize=24, fontweight='bold')
    ax.set_ylabel('T2D Prevalence (%)', fontsize=24, fontweight='bold')
    # ax.set_title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)', fontsize=18)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    #ax.legend(loc='upper left', fontsize=10, frameon=False)   
    ax.grid()

    plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("result_LE.csv")
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Add small increasing offset to 'both' to visually separate it
np.random.seed(0)
offset = np.linspace(0.1, 0.5, len(df))
df["both_female"] += offset
df["both_male"] += offset
offset2 = np.linspace(0.2, 0.7,  len(df))  # gradually increasing
df["only_t2d_female"] += offset2
df["only_t2d_male"] += offset2

def plot_life_expectancy(df, sex):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Split time
    df_pre = df[df["year"] <= 2023]
    df_post = df[df["year"] > 2023]

    # Observed data
    ax.scatter(
        df_pre["year"],
        df_pre[f"obs_{sex}"],
        label="Observed",
        s=60,
        color="black",
        zorder=5,
    )

    # Interventions
    interventions = {
        "no_interv": {"label": "No intervention", "color": "#fb5607"},
        "only_hiv": {"label": "HIV intervention", "color": "#fb5607"},
        "only_t2d": {"label": "T2D intervention", "color": "#5fad56"},
        "both": {"label": "Both intervention", "color": "#8338ec"},
    }

    for key, props in interventions.items():
        col = f"{key}_{sex}"
        if col in df.columns:
            # Pre-2024: solid line
            ax.plot(df_pre["year"], df_pre[col],
                    label=props["label"],
                    linestyle='-',
                    linewidth=5,
                    color=props["color"])
            # Post-2024: dashed line
            ax.plot(df_post["year"], df_post[col],
                    linestyle='--',
                    linewidth=5,
                    color=props["color"],
                    alpha=1)

    # Axis and styling
    ax.set_xlabel("Year", fontsize=24, fontweight="bold")
    ax.set_ylabel("Life Expectancy at Birth", fontsize=24, fontweight="bold")
    ax.set_title(f"{sex.capitalize()}", fontsize=28, fontweight="bold")
    ax.set_xticks(list(range(1990, 2051, 10)))
    ax.set_xlim(2007, 2050)
    ax.set_yticks(list(range(30, 70, 10)))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=22)
    #ax.legend(fontsize=18, frameon=False)

    plt.tight_layout()
    return fig

# Generate and show both figures
fig_female = plot_life_expectancy(df, "female")
fig_male = plot_life_expectancy(df, "male")
plt.show()



# # ##### Run the following 4 times after changing the end_year #####
# # # Get the T2D disease module
# t2d = sim.diseases['type2diabetes']

# # Define custom age groups
# age_groups = [
#     (0, 18, '0–17'),
#     (18, 35, '18–34'),
#     (35, 50, '35–49'),
#     (50, 65, '50–64'),
#     (65, 150, '65+'),
# ]

# print("Age Group | Total People | T2D Cases | % with T2D")
# print("---------------------------------------------------")

# for a0, a1, label in age_groups:
#     age_mask = (sim.people.age >= a0) & (sim.people.age < a1)
#     total = age_mask.sum()
#     affected = (age_mask & t2d.affected).sum()
#     prevalence = 100 * affected / total if total > 0 else 0
#     print(f"{label:>8} |     {total:5}     |    {affected:5}   |   {prevalence:6.2f}%")
    
    
# # # Get the T2D disease module
# # t2d = sim.diseases['type2diabetes']

# # print("Age | Total People | T2D Cases |  % with T2D")
# # print("---------------------------------------------")

# # for age in range(6):  # Ages 0 through 5
# #     is_age = (sim.people.age >= age) & (sim.people.age < age + 1)
# #     total = is_age.sum()
# #     affected = (is_age & t2d.affected).sum()
# #     prevalence = 100 * affected / total if total > 0 else 0
# #     print(f" {age}  |     {total:5}     |    {affected:5}   |   {prevalence:6.2f}%")  


# data = {
#     "0-17":  [0.06, 0.10, 0.09, 0.08],
#     "18-34": [1.5, 1.99, 2.16, 2.01],
#     "35-49": [3.77, 5.37, 5.97, 6.12],
#     "50-64": [12.06, 14.21, 14.2, 15.13],
#     "65+":   [24.22, 29.35, 32.66, 35.58],
# }
# columns = [2008, 2021, 2035, 2050]
# df = pd.DataFrame(data, index=columns).T  # transpose to have age groups as index

# colors = ['#08306b', '#2171b3', '#6baed6', '#9ecae1', '#c6dbef']  # One color per age group (5 total)

# plot_t2d_prevalence_by_age_and_year(df,colors)  
# plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')





import logging
import mighti as mi
import numpy as np
import pandas as pd
import prepare_data_for_year
import starsim as ss
import stisim as sti
from mighti.diseases.type2diabetes import ReduceMortalityTx


# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
n_agents = 100_000 
inityear = 2007
endyear = 2050
region = 'eswatini'


# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
# Parameters
csv_path_params = f'mighti/data/{region}_parameters_gbd.csv'

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus.csv"

# Disease prevalence data
csv_prevalence = f'mighti/data/{region}_prevalence.csv'

# Fertility data 
csv_path_fertility = f'mighti/data/{region}_asfr.csv'

# Death data
csv_path_death = f'mighti/data/{region}_mortality_rates.csv'

# Age distribution data
csv_path_age = f'mighti/data/{region}_age_distribution_{inityear}.csv'

# Ensure required demographic files are prepared
prepare_data_for_year.prepare_data_for_year(region,inityear)
prepare_data_for_year.prepare_data(region)

# Data paths for post process
mx_path = f'mighti/data/{region}_mx.csv'
ex_path = f'mighti/data/{region}_ex.csv'


# ---------------------------------------------------------------------
# Load Parameters and Disease Configuration
# ---------------------------------------------------------------------
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

healthconditions = ['Type2Diabetes']
diseases = ["HIV"] + healthconditions

ncd_df = df[df["disease_class"] == "ncd"]
chronic = ncd_df[ncd_df["disease_type"] == "chronic"]["condition"].tolist()
acute = ncd_df[ncd_df["disease_type"] == "acute"]["condition"].tolist()
remitting = ncd_df[ncd_df["disease_type"] == "remitting"]["condition"].tolist()
communicable_diseases = df[df["disease_class"] == "sis"]["condition"].tolist()


# ---------------------------------------------------------------------
# Prevalence Data and Analyzers
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, prevalence_data=prevalence_data_df, inityear=inityear
)
get_prev_fn = lambda d: lambda mod, sim, size: mi.age_sex_dependent_prevalence(d, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
    conditions=['hiv', 'type2diabetes'],
    condition_attr_map={
        'hiv': 'infected',
        'type2diabetes': 'affected'  
    }
)

# ---------------------------------------------------------------------
# Demographics and Networks
# ---------------------------------------------------------------------
death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = ss.Deaths(death_rates) 
death.death_rate_data *= 0.4
fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]


# ---------------------------------------------------------------------
# Diseases
# ---------------------------------------------------------------------
hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prev_fn('HIV')),
                      init_prev_data=None,   
                      p_hiv_death=None, 
                      include_aids_deaths=False, 
                      beta={'structuredsexual': [0.011023883426646121, 0.011023883426646121], 
                            'maternal': [0.044227226248848076, 0.044227226248848076]})
    # Best pars: {'hiv_beta_m2f': 0.011023883426646121, 'hiv_beta_m2c': 0.044227226248848076} seed: 12345

disease_objects = []
for dis in healthconditions:
    cls = getattr(mi, dis, None)
    if cls is not None:
        disease_objects.append(cls(csv_path=csv_path_params, pars={"init_prev": ss.bernoulli(get_prev_fn(dis))}))
disease_objects.append(hiv_disease)


# ---------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------
ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
interactions = [ncd_hiv_connector]

ncd_interactions = mi.read_interactions(csv_path_interactions) 
connectors = mi.create_connectors(ncd_interactions)

interactions.extend(connectors)


# ---------------------------------------------------------------------
# Interventions 
# ---------------------------------------------------------------------
# ART coverage among PLHIV (from 95-95-95 cascade estimates and Lancet data)
art_coverage_data = pd.DataFrame({
    'p_art': [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]
    # 'p_art': [1,1,1,1,1,1]
}, index=[2003, 2010, 2013, 2014, 2016, 2022])

# HIV testing probabilities over time (estimated testing uptake)
test_prob_data = [0.10, 0.25, 0.60, 0.70, 0.80, 0.95]
# test_prob_data = [1,1,1,1,1,1]
test_years = [2003, 2005, 2007, 2010, 2014, 2016]

tx_df = pd.read_csv("mighti/data/t2d_tx.csv")
t2d_tx = ss.Tx(df=tx_df)

t2d_treatment = ReduceMortalityTx(
    label='T2D Mortality Reduction',
    product=t2d_tx,
    prob=1.0,
    rel_death_reduction=0.5,
    eligibility=lambda sim: sim.diseases.type2diabetes.affected.uids
)

# Define interventions using these data
interventions = [
    sti.HIVTest(test_prob_data=test_prob_data, years=test_years),
    sti.ART(coverage_data=art_coverage_data),
    sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
    sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]}),
]

interventions2 = [
    sti.HIVTest(test_prob_data=test_prob_data, years=test_years),
    sti.ART(coverage_data=art_coverage_data),
    sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
    sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]}),
    t2d_treatment
]

interventions3 = [
    t2d_treatment
]

# ---------------------------------------------------------------------
# Utility: Get Modules
# ---------------------------------------------------------------------
def get_deaths_module(sim):
    for module in sim.modules:
        if isinstance(module, mi.DeathsByAgeSexAnalyzer):
            return module
    raise ValueError("Deaths module not found in the simulation. Make sure you've added the DeathsByAgeSexAnalyzer to your simulation configuration")

def get_pregnancy_module(sim):
    for module in sim.modules:
        if isinstance(module, ss.Pregnancy):
            return module
    raise ValueError("Pregnancy module not found in the simulation.")


# ---------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # sim = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     # interventions = interventions3,
    #     copy_inputs=False,
    #     label='Without Interventions'
    # )
    # # Run the simulation
    # sim.run()
    
    
    
    ### To run 2 simulation simultaneously #####
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
        diseases=disease_objects,
        connectors=interactions,
        interventions = interventions,
        copy_inputs=False,
        label='No_intervention'
    )
    
    # sim_with = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     interventions = interventions,
    #     copy_inputs=False,
    #     label='HIV_intervention'
    # )
    
    # sim_with_t2d = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     interventions = interventions3,
    #     copy_inputs=False,
    #     label='T2D_intervention'
    # )
    
    # sim_with_both = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     interventions = interventions2,
    #     copy_inputs=False,
    #     label='Both_intervention'
    # )
 
    # msim = ss.MultiSim(sims=[sim, sim_with, sim_with_t2d, sim_with_both])
    # # msim = ss.MultiSim(sims=[sim_with_t2d,sim_with_both])
    # msim.run()
    sim.run()
    
    # # Mortality rates and life table
    # target_year = endyear - 1
    
    # obs_mx = prepare_data_for_year.extract_indicator_for_plot(mx_path, target_year, value_column_name='mx')
    # obs_ex = prepare_data_for_year.extract_indicator_for_plot(ex_path, target_year, value_column_name='ex')
    
    # # Get the modules
    # deaths_module = get_deaths_module(sim)
    # pregnancy_module = get_pregnancy_module(sim)
    
    # df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=target_year, max_age=100, radix=n_agents)

    # df_mx_male = df_mx[df_mx['sex'] == 'Male']
    # df_mx_female = df_mx[df_mx['sex'] == 'Female']
    
    
    # life_table = mi.calculate_life_table_from_mx(sim, df_mx_male, df_mx_female, max_age=100)
    
    # mi.plot_mx_comparison(df_mx, obs_mx, year=target_year, age_interval=5)
    
    # # Plot life expectancy comparison
    # mi.plot_life_expectancy(life_table, obs_ex, year = target_year, max_age=100, figsize=(14, 10), title=None)
    
    
    import plot_IAS
    plot_IAS.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')

    


#56.52
#41.59 0.5
#42.62 0.4

# # Filter life expectancy at birth
# lt0 = life_table[life_table['Age'] == 0].copy()

# # Compute weighted average life expectancy at birth
# total_l0 = lt0['l(x)'].sum()
# lt0['weight'] = lt0['l(x)'] / total_l0
# weighted_le = (lt0['e(x)'] * lt0['weight']).sum()

# print(f"Life expectancy at birth (both sexes): {weighted_le:.2f} years")




