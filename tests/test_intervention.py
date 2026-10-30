"""
Test that a treatment reducing mortality reduces deaths due to Type 2 Diabetes.

Runs two simulations:
- one with T2D only
- one with T2D and a treatment that halves T2D mortality while held (an ss.Treatment delivered by ss.treat)

Asserts that total deaths from T2D are lower with the treatment.
"""

import sciris as sc
import starsim as ss
import mighti as mi
import os


# Settings
do_plot = False
sc.options(interactive=do_plot)

# File path to parameter file
thisdir = os.path.dirname(__file__)
param_path = os.path.join(thisdir, 'test_data', 'eswatini_parameters.csv')


def test_reduce_mortality_tx_runs():

    def make_sim(treat=False):
        t2d = mi.diseases.Type2Diabetes(csv_path=param_path, pars={'init_prev': ss.bernoulli(0.9), 'p_death': 0.1})
        interventions = []
        if treat: # Everyone with T2D is treated, which halves their mortality from it while they hold the treatment
            tx = ss.Treatment(efficacy=dict(death=0.5), name='t2d_tx')
            interventions = ss.treat(tx, where=t2d.affected, interval=ss.years(1), label='T2D mortality reduction')
        return ss.Sim(start=2000, stop=2050, dt=1, n_agents=10000, diseases=t2d, interventions=interventions)

    sim0 = make_sim().run()
    sim1 = make_sim(treat=True).run()

    # Compare total T2D deaths
    d0 = sim0.results.type2diabetes.death.values.sum()
    d1 = sim1.results.type2diabetes.death.values.sum()
    assert d1 < d0, f'T2D deaths should be lower with treatment: {d1=} < {d0=}'
    print(f'✓ T2D deaths reduced from {d0:.0f} → {d1:.0f} by treatment')


if __name__ == '__main__':
    test_reduce_mortality_tx_runs()
