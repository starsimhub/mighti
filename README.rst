Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)
=============================================================================

**Warning!** MIGHTI is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is *not yet ready* to be used for real research or policy analysis without careful validation.

MIGHTI (Modeling Integrated Generational Health, Transmission, and Interventions) is an agent-based modeling framework designed to simulate the dynamics of infectious diseases, non-communicable diseases (NCDs), and their interactions over time. MIGHTI builds on the Starsim architecture.

Requirements
------------

- Python 3.9–3.13

We recommend, but do not require, installing MIGHTI in a virtual environment such as `Anaconda <https://www.anaconda.com/>`_.

Installation
------------

MIGHTI is **not yet available on PyPI**, but you can install it directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/starsimhub/mighti.git

Alternatively, you can clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/starsimhub/mighti.git
    cd mighti
    pip install -e .


Please also make sure to install its dependencies:

.. code-block:: bash

    pip install starsim stisim


Usage and Documentation
------------------------

Documentation and examples are currently under development.

For help getting started, please see the following example scripts:

- ``mighti_demography.py`` — runs demography-related modules (e.g., life expectancy, mortality)
- ``mighti_main.py`` — runs full disease-related simulations
