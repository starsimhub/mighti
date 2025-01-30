import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os


# ####### Demographic pyramid #######
# # Load the data
# file_path = 'tests/test_data/Demography_Eswatini_2023.csv'  # Update this if needed
# df = pd.read_csv(file_path)

# # Ensure column names match your file
# df.columns = ['age_group', 'male', 'female']

# # Convert population counts to thousands (keep male as negative for left alignment)
# df['male'] = -df['male']
# df['female'] = df['female']

# # Plot the demographic pyramid
# fig, ax = plt.subplots(figsize=(10, 6))

# ax.barh(df['age_group'], df['male'], color='blue', label='Male', alpha=0.7)
# ax.barh(df['age_group'], df['female'], color='red', label='Female', alpha=0.7)

# # Format plot
# ax.set_xlabel('Population (Thousands)')
# ax.set_ylabel('Age Group')
# # ax.set_title('Demographic Pyramid of Eswatini (2023)')
# ax.legend(loc='upper right')

# # Flip y-axis so youngest age groups appear at the bottom
# ax.invert_yaxis()

# # Set X-axis to be symmetric and rounded to nearest multiple of 20
# max_population = max(df['female'].max(), -df['male'].min())
# max_tick = ((max_population // 20) + 1) * 20  # Round up to the nearest multiple of 20
# ax.set_xlim(-max_tick, max_tick)

# # Define custom ticks at 20, 40, 60, 80
# ax.set_xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
# ax.set_xticklabels(['80', '60', '40', '20', '0', '20', '40', '60', '80'])

# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.show()


# ####### Fertility #######
# # Load the ASFR data
# file_path = 'tests/test_data/eswatini_asfr.csv'  # Update with actual path
# df = pd.read_csv(file_path)

# # Filter for the year 2023
# df_2023 = df[df['Time'] == 2023]

# # Plot the fertility rate by age
# plt.figure(figsize=(8, 5))
# plt.plot(df_2023['AgeGrp'], df_2023['ASFR'], marker='o', linestyle='-', color='black', label='ASFR (2023)')

# # Formatting
# plt.xlabel('Age')
# plt.ylabel('Age-Specific Fertility Rate (per 1,000 women)')
# # plt.title('Age-Specific Fertility Rate in Eswatini (2023)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(df_2023['AgeGrp'])  # Ensure all age groups appear
# plt.legend()

# # Show plot
# plt.show()


# #######MOrtality #######
# # Load the death rate data
# file_path = 'tests/test_data/eswatini_deaths.csv'  # Update with actual path
# df = pd.read_csv(file_path)

# # Filter for the year 2020
# df_2023 = df[df['Time'] == 2020]  

# # Separate data for males and females
# df_male = df_2023[df_2023['Sex'] == 'Male']
# df_female = df_2023[df_2023['Sex'] == 'Female']

# # Plot the death rate by age for males and females
# plt.figure(figsize=(8, 5))
# plt.plot(df_male['AgeGrpStart'], df_male['mx'], marker='o', linestyle='-', color='blue', label='Male Death Rate')
# plt.plot(df_female['AgeGrpStart'], df_female['mx'], marker='o', linestyle='-', color='red', label='Female Death Rate')

# # Formatting
# plt.xlabel('Age')
# plt.ylabel('Death Rate (per person)')
# # plt.title('Age-Specific Death Rate in Eswatini (2023)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(df_2023['AgeGrpStart'].unique())  # Ensure all age groups appear
# plt.legend()

# # Show plot
# plt.show()


# ####### Prevalence HIV, T2D #######
# # Load the prevalence data
# file_path = 'mighti/data/prevalence_data_eswatini.csv'  # Update with actual path
# df = pd.read_csv(file_path)

# # Filter for the year 2021
# df_2021 = df[df['Year'] == 2021]

# # Extract relevant columns
# df_2021 = df_2021[['Age', 'HIV_male', 'HIV_female', 'Type2Diabetes_male', 'Type2Diabetes_female']]

# # Convert to percentages
# df_2021[['HIV_male', 'HIV_female', 'Type2Diabetes_male', 'Type2Diabetes_female']] *= 100

# # Bar width
# bar_width = 0.4
# age_groups = df_2021['Age']

# # Set x positions for bars
# x = np.arange(len(age_groups))

# # Create figure and subplots
# fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# # Plot HIV prevalence
# axs[0].bar(x - bar_width/2, df_2021['HIV_male'], width=bar_width, color='blue', label='HIV Male')
# axs[0].bar(x + bar_width/2, df_2021['HIV_female'], width=bar_width, color='red', label='HIV Female')
# axs[0].set_xticks(x)
# axs[0].set_xticklabels(age_groups, rotation=45)
# axs[0].set_xlabel('Age')
# axs[0].set_ylabel('Prevalence (%)')
# axs[0].set_title('HIV Prevalence by Age (2021)')
# axs[0].legend()
# axs[0].grid(axis='y', linestyle='--', alpha=0.6)

# # Plot T2D prevalence
# axs[1].bar(x - bar_width/2, df_2021['Type2Diabetes_male'], width=bar_width, color='blue', label='T2D Male')
# axs[1].bar(x + bar_width/2, df_2021['Type2Diabetes_female'], width=bar_width, color='red', label='T2D Female')
# axs[1].set_xticks(x)
# axs[1].set_xticklabels(age_groups, rotation=45)
# axs[1].set_xlabel('Age')
# axs[1].set_title('Type 2 Diabetes Prevalence by Age (2021)')
# axs[1].legend()
# axs[1].grid(axis='y', linestyle='--', alpha=0.6)

# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()


