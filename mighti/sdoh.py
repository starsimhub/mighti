import starsim as ss
import numpy as np

class HousingSituation(ss.Module):
    def __init__(self):
        super().__init__()
        self.name = 'housing_situation'
        self.housing_unstable = ss.State(name='housing_unstable', label='Unstable Housing')

    def initialize(self, sim):
        self.sim = sim
        self.people = sim.people

        # Link the state to people
        self.housing_unstable.link_people(self.people)

        # Allocate memory for raw array
        n = len(self.people.uid)
        self.housing_unstable.len_tot = n
        self.housing_unstable.len_used = n
        self.housing_unstable.raw = np.full(n, fill_value=self.housing_unstable.nan, dtype=self.housing_unstable.dtype)

        # Initialize values
        self.housing_unstable.init_vals()
        self.housing_unstable.set(self.people.uid, new_vals=np.random.rand(n) < 0.3)

    def step(self, sim):  # ✅ method must be named `step`, not `update`
        # Example dynamics (only if `employed` is defined)
        if hasattr(self.people, 'employed'):
            employed = self.people.employed
            at_risk = self.housing_unstable & employed
            to_stabilize = at_risk[np.random.rand(len(at_risk)) < 0.3]
            self.housing_unstable[to_stabilize] = False

            not_employed = ~employed
            newly_unstable = not_employed & ~self.housing_unstable
            to_unhouse = newly_unstable[np.random.rand(len(newly_unstable)) < 0.1]
            self.housing_unstable[to_unhouse] = True