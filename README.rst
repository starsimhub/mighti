Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)
=======================================

**Warning!** MIGHTI is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is not yet ready to be used for real research or policy analysis without careful validation.

MIGHTI (Modeling Integrated Generational Health, Transmission, and Interventions) is an agent-based modeling framework designed to simulate the dynamics of infectious diseases, non-communicable diseases, and their interactions over time. MIGHTI builds on the Starsim architecture.

=======================================
Requirements

* Python 3.9–3.13

We recommend, but do not require, installing MIGHTI in a virtual environment such as [Anaconda](https://www.anaconda.com/).

=======================================
Installation

MIGHTI is **not yet available on PyPI**, but you can install it locally via GitHub:

```bash
pip install git+https://github.com/starsimhub/mighti.git
```

Alternatively, you can clone the repository and install in editable mode:

```bash
git clone https://github.com/starsimhub/mighti.git
cd mighti
pip install -e .
```

=======================================
Usage and Documentation

Documentation and examples are currently under development. 

For help getting started, please see:

``mighti_demography.py``: to run demography-related modules (e.g., life expectancy, mortality)

``mighti_main.py``: to run disease-related simulations


