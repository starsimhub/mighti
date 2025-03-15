import starsim as ss
import mighti as mi
import sciris as sc
import matplotlib.pyplot as plt


# # Specific connector for the interaction between HIV and Type2Diabetes
# class HIVType2DiabetesConnector(ss.Connector):
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Type2Diabetes')
#         self.define_pars(rel_sus=1.95)
#         self.update_pars(pars, **kwargs)
#         self.time = sc.autolist()
#         self.rel_sus = sc.autolist()
#         self.t2d_prev = sc.autolist()
#         self.hiv_prev = sc.autolist()
        
#     def step(self):
#         t2d = self.sim.diseases.type2diabetes
#         hiv = self.sim.diseases.hiv
#         hiv.rel_sus[t2d.affected.uids] = self.pars.rel_sus
        
#         # Collecting data for analysis
#         self.time += self.sim.t
#         self.rel_sus += hiv.rel_sus.mean()
#         self.t2d_prev += t2d.results.prevalence[self.sim.ti]
#         self.hiv_prev += hiv.results.prevalence[self.sim.ti]
#         return

# Specific connector for the interaction between HIV and Type2Diabetes
class HIVType2DiabetesConnector(ss.Connector):
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Type2Diabetes')
        self.define_pars(rel_sus=1.95)
        self.update_pars(pars, **kwargs)
        self.time = sc.autolist()
        self.rel_sus = sc.autolist()
        self.t2d_prev = sc.autolist()
        self.hiv_prev = sc.autolist()
        
    def step(self):
        t2d = self.sim.diseases.type2diabetes
        hiv = self.sim.diseases.hiv
        t2d.rel_sus[hiv.infected.uids] = self.pars.rel_sus
        
        # Collecting data for analysis
        self.time += self.sim.t
        self.rel_sus += t2d.rel_sus.mean()
        self.t2d_prev += t2d.results.prevalence[self.sim.ti]
        self.hiv_prev += hiv.results.prevalence[self.sim.ti]
        return