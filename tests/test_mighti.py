"""
Preliminary demo script for running MIGHTI analyses
"""

# Imports
import starsim as ss
import mighti as mi
import pylab as pl
import pandas as pd

# Create a figure for viewing results
pl.figure()

# Create the disease list
hiv = ss.HIV(
    beta={
        'mf': [0.0008, 0.0004],  # Per-act transmission probability from sexual contacts
        'maternal': [0.2, 0]},   # MTCT probability
)
depression = mi.Depression()
diseases = [hiv, depression]

# Create the networks - sexual and maternal
mf = ss.MFNet(
    duration=1/24,  # Mean duration of relationships
    acts=80,
)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)

# Create a simulation with the depression module from conditions.py, Starsim's default HIV module,
# and a connector as defined in interactions.py. Vary the relative risk of HIV acquisition that
# is associated with depression
for rel_risk in [1, 3]:
    sim = ss.Sim(
        n_agents=5_000,
        networks=networks,
        diseases=diseases,
        connectors=mi.hiv_depression(rel_sus_hiv_depression=rel_risk)
    )
    sim.run()
    pl.plot(sim.yearvec, sim.results.hiv.n_infected, label=f'Relative risk={rel_risk}')

pl.title('HIV infections')
pl.xlabel('Year')
pl.ylabel('Count')
pl.legend()
pl.show()


