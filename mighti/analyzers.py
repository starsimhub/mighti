import numpy as np
import pandas as pd
import starsim as ss

__all__ = ["DeathsByAgeSexAnalyzer", "SurvivorshipAnalyzer", "ConditionAtDeathAnalyzer"]

class DeathsByAgeSexAnalyzer(ss.Analyzer):

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('infant_deaths', label='Cumulative infant deaths', dtype=int),
            ss.Result('male_deaths_by_age', label='Number of male deaths by age', dtype=int, shape=101, ),
            ss.Result('female_deaths_by_age', label='Number of female deaths by age', dtype=int, shape=101, )
        )
        return

    def step(self):
        people = self.sim.people
        ti = self.sim.ti

        self.results.infant_deaths[ti] = len(people.dead[people.age < 1])

        for uid in people.dead.uids:
            age = int(min(people.age[uid], 100))            
            if people.female[uid]:
                self.results.female_deaths_by_age[age] += 1
            else:
                self.results.male_deaths_by_age[age] += 1
              
                
class SurvivorshipAnalyzer(ss.Analyzer):

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = 'survivorship_analyzer'

        self.max_age = max_age
        self.survivorship_data = {'Male': np.zeros(max_age), 'Female': np.zeros(max_age)}

    # def init_post(self):
    #     self.survivorship_data = np.zeros(shape=(self.max_age, 2))

    def step(self):
        ppl = self.sim.people
        for age in range(self.max_age):
            for sex in ['Male', 'Female']:
                self.survivorship_data[sex][age] += len(ppl.age[(ppl.age >= age) & (ppl.age < age+1) & (ppl.female == (sex=='Female'))])


class ConditionAtDeathAnalyzer(ss.Analyzer):
    def __init__(self, conditions=None, condition_attr_map=None, **kwargs):
        super().__init__(**kwargs)
        self.conditions = [c.lower() for c in (conditions or [])]
        self.condition_attr_map = condition_attr_map or {}
        self.records = []
        self.condition_snapshots = {}  # (uid, condition) → True/False

    def init_results(self):
        super().init_results()
        self.records = []
        self.condition_snapshots = {}

    def step(self):
        ppl = self.sim.people
        ti = self.sim.ti
        year = self.sim.t.yearvec[ti]

        # print(f"\n[ConditionAtDeathAnalyzer] Step {ti}, Year {year}")
        # print(f"Number of deaths this step: {len(ppl.dead.uids)}")
        
        for uid in ppl.dead.uids:
            record = {
                'uid': uid,
                'year': year,
                'age': ppl.age[uid],
                'sex': 'Female' if ppl.female[uid] else 'Male',
            }
        
        for cond in self.conditions:
            # ti_dead_val = ppl[cond].ti_dead[uid]
        
            # if not np.isnan(ti_dead_val):
            #     ti_dead_idx = int(ti_dead_val)
            #     if ti_dead_idx < len(self.sim.diseases[cond].t.abstvec):
            #         condition_ti = self.sim.diseases[cond].t.abstvec[ti_dead_idx]
            #         died_of_cond = (condition_ti > ti - 1) and (condition_ti <= ti)
            #     else:
            #         died_of_cond = False  # dead, but beyond current abstvec — skip tagging
            # else:
            #     died_of_cond = False
            ti_dead = ppl[cond].ti_dead[uid]
            if not np.isnan(ti_dead):
                condition_ti = self.sim.diseases[cond].t.abstvec[int(ti_dead)]
                died_of_cond = (condition_ti > ti - 1) and (condition_ti <= ti)
            else:
                condition_ti = np.nan
                died_of_cond = False
        
            record[f'died_{cond}'] = died_of_cond
        
                # print(f"UID {uid}: condition={cond}, ti_dead={ppl[cond].ti_dead[uid]}, condition_ti={condition_ti if not np.isnan(ppl[cond].ti_dead[uid]) else 'nan'}, died_{cond}={died_of_cond}")
        
            self.records.append(record)
            
    def to_df(self):
        return pd.DataFrame(self.records)
                