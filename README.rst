Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)
=============================================================================

**Warning!** MIGHTI is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is *not yet ready* to be used for real research or policy analysis without careful validation.

MIGHTI is an agent-based modeling framework designed to simulate the dynamics of infectious diseases, non-communicable diseases (NCDs), and their interactions over time. MIGHTI builds on the Starsim architecture.

**Documentation:** See the `MIGHTI Wiki <https://github.com/starsimhub/mighti/wiki>`_ for usage instructions and examples.

Requirements
------------

- Python 3.9–3.13

We recommend, but do not require, installing MIGHTI in a virtual environment such as `Anaconda <https://www.anaconda.com/>`_.

Installation
------------

MIGHTI is **not yet available on PyPI**, but you can install it directly from GitHub:

.. code-block:: bash

    git clone https://github.com/starsimhub/mighti.git
    cd mighti
    pip install -e .


Please also make sure to install its dependencies:

.. code-block:: bash

    pip install starsim stisim


Running an Example
------------

python mighti_main.py

This will run a sample simulation that includes demography, HIV, and NCD modules. Outputs will be saved in the outputs/ folder.

You can also run:
	•	mighti_demography.py — mortality and life expectancy module
	•	mighti_calibration.py — Optuna-based parameter calibration


Usage and Documentation
------------

MIGHTI is based on Starsim, please refer to `Starsim documentation <https://docs.idmod.org/projects/starsim/en/latest/>`_ for additional information.


References
------------

MIGHTI uses data from:
	•	Demography: UN WPP 2024
	•	NCDs: Global Burden of Disease (GBD)
	•	HIV: DHS, SHIMS


Contributing
------------

Contributions to the MIGHTI project are welcome! Please read `CONTRIBUTING.mst <https://github.com/starsimhub/mighti/blob/main/contributing.rst>`_ for details on our code of conduct, and the process for submitting pull requests.


Disclaimer
------------

This code was developed by researchers at NYU, IDM, and collaborators. It is shared under the MIT License to foster reproducibility and future development. No guarantees are made regarding functionality or support. You are free to fork and modify the code under the terms of the license.


References
------------------------

The MIGHTI framework incorporates data from the following public sources:

	•	Demography data:
		World Population Prospects 2024
		https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality

	•	Disease data:
		Global Burden of Disease Study (GBD)
		https://vizhub.healthdata.org/gbd-results/

	•	HIV data:
		•	Demographic and Health Surveys (DHS) https://dhsprogram.com/pubs/pdf/FR202/FR202.pdf

		•	Swaziland HIV Incidence Measurement Survey (SHIMS) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5291824/
