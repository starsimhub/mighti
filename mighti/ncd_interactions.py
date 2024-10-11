from collections import defaultdict
import pandas as pd
import starsim as ss
import mighti as mi  # Assuming NCDs are defined her


__all__ = ['obesity_diabetes2','depression_obesity','hypertension_heart']

# Function to read in data with age and sex dependency
def read_interactions(datafile=None):
    """
    Read in datafile with risk/condition interactions including age and sex.
    """
    if datafile is None:
        datafile = 'mighti/data/interactions_ncds.csv'  # Adjust the path to your CSV file

    df = pd.read_csv(datafile)
    
    # The structure to hold the interaction data
    rel_sus = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Iterate over rows and store relative risk by age and sex
    for _, row in df.iterrows():
        risk_factor = row['Risk.factor']
        condition = row['variable']
        age = row['Age']
        sex = row['Sex']
        value = row['value']  # This is the relative risk

        # Store relative risk in the dictionary based on risk_factor, condition, age, and sex
        rel_sus[risk_factor][condition][age][sex] = value

    return rel_sus


class NCDInteractions:
    def __init__(self, interaction_data_file, age_bins):
        # Load the interaction data from a CSV file
        self.rel_sus = read_interactions(interaction_data_file)
        self.age_bins = age_bins  # Store the age bins

    def get_interaction(self, risk_factor, condition, age, sex):
        # Map the person's age to the appropriate age bin
        age_bin = self.get_age_bin(age)
        
        if age_bin is None:
            return 1.0
        
        # Fetch the relative risk from the interaction dictionary
        try:
            rr = self.rel_sus[risk_factor][condition][age_bin][sex]
            return rr
        except KeyError:
            # print(f"Age bin {age_bin} or other key not found for Risk factor: {risk_factor}, Condition: {condition}, Sex: {sex}")
            return 1.0  # Default OR/RR if no interaction is found
    
    def get_age_bin(self, age):
        # Debugging age bin mapping
        for i in range(len(self.age_bins) - 1):
            if self.age_bins[i] <= age < self.age_bins[i + 1]:
                return self.age_bins[i]  # Return the lower bound of the bin
        return None  # Return None if no bin is found
    

class obesity_diabetes2(ss.Connector):
    """Obesity increases the risk of developing Type2Diabetes based on age and sex."""
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='Obesity-Type2Diabetes', requires=[mi.Obesity, mi.Type2Diabetes])
        
        # Load interaction data from CSV
        self.ncd_interactions = NCDInteractions('mighti/data/interactions_ncds.csv', age_bins=[0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        
        # Get UIDs of people affected by obesity
        affected_uids = sim.diseases.obesity.affected.uids
        susceptible_uids = sim.diseases.type2diabetes.susceptible.uids
        print(f"Number of susceptible individuals: {len(susceptible_uids)}")
        
        # Loop through affected individuals and adjust their likelihood of developing Type2Diabetes
        for uid in affected_uids:
            person = sim.people[int(uid)]
            
            # Determine the person's sex using the 'female' attribute
            person_sex = 'Female' if person.female else 'Male'
            
            # Get the age and sex-dependent relative risk from the CSV
            relative_risk = self.ncd_interactions.get_interaction('Obesity', 'Type2Diabetes', person.age, person_sex)
    
            if sim.diseases.type2diabetes.susceptible[person.uid]:
                print(f"Person {person.uid} is susceptible to Type2Diabetes.")
                
                # Instead of manually applying the relative risk, we now pass it to `make_new_cases`
                print(f"Applying relative risk adjustment of {relative_risk} to the probability of developing Type2Diabetes.")
                
                # Call `make_new_cases` with the relative risk adjustment
                sim.diseases.type2diabetes.make_new_cases(relative_risk=relative_risk)
    
        return

# # List all NCD interaction classes for external visibility
# __all__ = [
#     'obesity_diabetes2', 'depression_obesity', 'hypertension_heart', 
# ]

# # Define a class to load and query interaction data from a CSV file
# class NCDInteractions:
#     def __init__(self, interaction_data_file):
#         # Load the interaction data from a CSV file
#         self.data = pd.read_csv(interaction_data_file)

#     def get_interaction(self, risk_factor, condition, age, sex):
#         # Extract the odds ratio (or relative risk) for interaction based on the age and sex
#         subset = self.data[
#             (self.data['Risk.factor'] == risk_factor) &
#             (self.data['variable'] == condition) &
#             (self.data['Age'] == age) &
#             (self.data['Sex'] == sex)
#         ]
#         if not subset.empty:
#             return subset['value'].values[0]  # The 'value' column contains the OR/RR
#         else:
#             return 1.0  # Default OR/RR if no interaction is found

# # General NCD interaction class (used when a specific interaction is not defined)
# class GeneralNCDConnector(ss.Connector):
#     def __init__(self, risk_factor, condition, pars=None, **kwargs):
#         label = f'{risk_factor}-{condition}'
#         requires = [getattr(mi, risk_factor), getattr(mi, condition)]  # Dynamically get NCD classes
#         super().__init__(label=label, requires=requires)
#         self.risk_factor = risk_factor
#         self.condition = condition
#         self.update_pars(pars, **kwargs)

#     def update(self):
#         sim = self.sim
#         ncd_interactions = NCDInteractions('mighti/data/interactions_ncds.csv')  # Load CSV

#         for person in sim.people:
#             # Determine the person's sex using the 'female' attribute
#             person_sex = 'Female' if person.female else 'Male'
            
#             # Check if the person is affected by the risk factor
#             if getattr(sim.diseases, self.risk_factor.lower()).affected[person.uid]:
#                 # Get the relative risk based on their age and sex
#                 relative_risk = ncd_interactions.get_interaction(
#                     self.risk_factor, self.condition, person.age, person_sex
#                 )
#                 # Apply the relative risk to the susceptibility of the second condition
#                 getattr(sim.diseases, self.condition.lower()).rel_sus[person.uid] *= relative_risk
#         return

# # Specific NCD interaction class: Obesity and Type 2 Diabetes
# class obesity_diabetes2(ss.Connector):
#     """Obesity increases the risk of developing Type2Diabetes based on age and sex."""
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='Obesity-Type2Diabetes', requires=[mi.Obesity, mi.Type2Diabetes])
        
#         # Load interaction data from CSV
#         self.ncd_interactions = NCDInteractions('mighti/data/interactions_ncds.csv')
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
        
#         # Get UIDs of people affected by obesity
#         affected_uids = sim.diseases.obesity.affected.uids
        
#         # Loop through affected individuals and adjust their likelihood of developing Type2Diabetes
#         for uid in affected_uids:
#             person = sim.people[int(uid)] 
            
#             # Determine the person's sex using the 'female' attribute
#             person_sex = 'Female' if person.female else 'Male'
            
#             # Get the age and sex-dependent relative risk from the CSV
#             relative_risk = self.ncd_interactions.get_interaction('Obesity', 'Type2Diabetes', person.age, person_sex)
            

#             if sim.diseases.obesity.affected[person.uid]: 
#                 # Apply the relative risk to the incidence (make_new_cases likelihood)
#                 sim.diseases.type2diabetes.rel_sus[person.uid] *= relative_risk
#         return
    
    
# class obesity_diabetes2(ss.Connector):
#     def __init__(self, pars=None, **kwargs):
        
        
#         super().__init__(label='Obesity-Type2Diabetes', requires=[mi.Obesity, mi.Type2Diabetes])
#         self.update_pars(pars, **kwargs)

#     def update(self):
#         sim = self.sim
#         ncd_interactions = NCDInteractions('mighti/data/interactions_ncds.csv')  # Load CSV

#         for person in sim.people:
#             # Access the affected status through sim.diseases
#             if sim.diseases.obesity.affected[person.uid]:  # Use the disease object to access 'affected' status
#                 # Get the relative risk from the CSV based on the person's age and sex
#                 relative_risk = ncd_interactions.get_interaction('Obesity', 'Type2Diabetes', person.age, person.sex)
#                 sim.diseases.type2diabetes.rel_sus[person.uid] *= relative_risk
#         return
    
    

# class hiv_hypertension(ss.Connector):
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Hypertension', requires=[ss.HIV, mi.Hypertension])

        
#         self.default_pars(
#             rel_sus_hiv_hypertension=1.3,  # People with hypertension are 1.3x more likely to acquire HIV
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with hypertension
#         sim.diseases.hiv.rel_sus[sim.people.hypertension.affected] = self.pars.rel_sus_hiv_hypertension
#         return

    

class depression_obesity(ss.Connector):
    """Depression increases the risk of developing obesity based on age and sex"""
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='Depression-Obesity', requires=[mi.Depression, mi.Obesity])
        self.update_pars(pars, **kwargs)

    def update(self):
        sim = self.sim
        ncd_interactions = NCDInteractions('mighti/data/interactions_ncds.csv')  # Load CSV
        
        for person in sim.people:
            if person.depression.affected:
                # Get the relative riskfrom the CSV based on the person's age and sex
                relative_risk = ncd_interactions.get_interaction('Depression', 'Obesity', person.age, person.sex)
                sim.diseases.obesity.rel_sus[person.uid] *= relative_risk
        return

class hypertension_heart(ss.Connector):
    """Hypertension increases the risk of heart diseases based on age and sex"""
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='Hypertension-Heart', requires=[mi.Hypertension, mi.HeartDisease])
        self.update_pars(pars, **kwargs)

    def update(self):
        sim = self.sim
        ncd_interactions = NCDInteractions('mighti/data/interactions_ncds.csv')  # Load CSV
        
        for person in sim.people:
            if person.hypertension.affected:
                # Get the relative riskfrom the CSV based on the person's age and sex
                relative_risk = ncd_interactions.get_interaction('Hypertension', 'HeartDisease', person.age, person.sex)
                sim.diseases.heartdisease.rel_sus[person.uid] *= relative_risk
        return

# You can continue defining specific classes as needed for different NCD combinations.