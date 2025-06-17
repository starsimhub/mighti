import starsim as ss
import numpy as np
import matplotlib.pyplot as plt
from mighti.sdoh import HousingSituation

def test_housing_initialization():
    n = 1000
    ppl = ss.People(n)
    housing = HousingSituation(prob=0.4)
    sim = ss.Sim(people=ppl, modules=[housing], start=2000, stop=2001)
    
    sim.initialize()  # Only init, don't run
    unstable = housing.housing_unstable.values

    # Check array length and proportion
    assert len(unstable) == n
    prop = unstable.mean()
    assert 0.2 < prop < 0.6, f"Unexpected proportion: {prop}"

def test_housing_dynamic_update():
    n = 1000
    ppl = ss.People(n)
    ppl.employed = ss.BoolArr(name='employed')
    ppl.employed.link_people(ppl)
    ppl.employed.init_vals()
    ppl.employed[:] = np.random.rand(n) < 0.5  # 50% employed

    housing = HousingSituation(prob=0.8)  # Make most people unstable
    sim = ss.Sim(people=ppl, modules=[housing], start=2000, stop=2001)
    
    sim.initialize()
    unstable_before = housing.housing_unstable.count()
    sim.step()  # Step once
    unstable_after = housing.housing_unstable.count()

    assert unstable_after < unstable_before, "Housing instability not reduced"
    

if __name__ == '__main__':
    n_agents = 1000
    housing = HousingSituation(prob=0.4)
    people = ss.People(n_agents)
    sim = ss.Sim(people=people, modules=[housing], start=2000, stop=2010)

    # Track instability over time
    housing_instability = []

    sim.initialize()
    for _ in sim.yearvec:
        sim.step()
        prop_unstable = housing.housing_unstable.values.mean()
        housing_instability.append(prop_unstable)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(sim.yearvec, housing_instability, marker='o')
    plt.title("Housing Instability Over Time")
    plt.xlabel("Year")
    plt.ylabel("Proportion Unstable")
    plt.grid(True)
    plt.tight_layout()
    plt.show()