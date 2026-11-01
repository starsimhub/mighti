"""
Social determinants of health (SDoH): a yes/no property of each person, inherited from the mother at birth
"""

import numpy as np
import pandas as pd
import starsim as ss

__all__ = ['BaseSDoH']


class BaseSDoH(ss.Module):
    """
    A yes/no social determinant of health, as a property of each person (e.g. `people.neighbourhood_situation`)

    At the start, each person has it with probability `p_stable` (a little less for children: times
    0.9 + 0.1/(1 + exp(-(age - 15)/5))). A newborn takes the mother's value with probability `inherit_prob`, and
    otherwise has it with probability `p_stable`. Each year, people with it lose it at an annual probability
    `p_loss`, and people without it gain it at `p_gain`.

    Args:
        name (str): the name of the module, and of its row in the CSV file (column `state`)
        csv_path (str): optional CSV file with columns `state`, `state_prob` (`p_stable`), and `inherit_prob`
        condition_name (str): the name of its row in the CSV file, if not `name`
        default_p_stable (float): `p_stable` if not in the file
        default_inherit_prob (float): `inherit_prob` if not in the file
        state_attr (str): the name of the property of the people (default: `<name>_situation`)
        p_loss (float): annual probability of losing it
        p_gain (float): annual probability of gaining it
    """
    def __init__(self, name, csv_path=None, condition_name=None, default_p_stable=0.7, default_inherit_prob=0.9, state_attr=None, p_loss=0.01, p_gain=0.10):
        super().__init__(name=name)
        self.state_attr = state_attr or f'{name}_situation'
        p_stable, inherit_prob = default_p_stable, default_inherit_prob
        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            row = df[df['state'].str.strip().str.lower() == (condition_name or name).lower()]
            if not row.empty:
                if pd.notna(row['state_prob'].iloc[0]):
                    p_stable = float(row['state_prob'].iloc[0])
                if pd.notna(row['inherit_prob'].iloc[0]):
                    inherit_prob = float(row['inherit_prob'].iloc[0])
        self.define_pars(
            p_stable = p_stable,
            inherit_prob = ss.bernoulli(p=inherit_prob),
            p_init = ss.bernoulli(p=0), # Set in init_post()
            p_newborn = ss.bernoulli(p=p_stable),
            p_loss = ss.bernoulli(p=ss.probperyear(p_loss)),
            p_gain = ss.bernoulli(p=ss.probperyear(p_gain)),
        )
        return

    @property
    def p_stable(self):
        return self.pars.p_stable

    @property
    def inherit_prob(self):
        return self.pars.inherit_prob.pars.p

    def init_pre(self, sim):
        super().init_pre(sim)
        self.state = sim.people.add_state(ss.BoolArr(self.state_attr))
        preg = sim.get_module(ss.Pregnancy, die=False)
        if preg is not None:
            preg.add_callback(preg.birth, self.on_birth)
        return

    def init_post(self):
        """ The initial values, by age """
        super().init_post()
        ppl = self.sim.people
        uids = ppl.auids
        age_mod = 0.9 + 0.1/(1 + np.exp(-(ppl.age[uids] - 15)/5))
        self.pars.p_init.set(p=np.clip(self.pars.p_stable*age_mod, 0.05, 0.99))
        self.state[uids] = False
        self.state[self.pars.p_init.filter(uids)] = True
        return

    def on_birth(self, uids):
        """ Newborns inherit their mother's value, or draw their own """
        inherit, own = self.pars.inherit_prob.filter(uids, both=True)
        mothers = self.sim.people.parent[inherit]
        self.state[inherit] = self.state[ss.uids(mothers)]
        self.state[own] = False
        self.state[self.pars.p_newborn.filter(own)] = True
        return

    def step(self):
        """ Losing and gaining it """
        state = self.state
        lose = self.pars.p_loss.filter(state.uids)
        gain = self.pars.p_gain.filter((~state).uids)
        state[lose] = False
        state[gain] = True
        return

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('prop_stable', dtype=float, scale=False, label=f'Share with {self.state_attr}'))
        return

    def update_results(self):
        super().update_results()
        self.results.prop_stable[self.ti] = self.state.mean()
        return
