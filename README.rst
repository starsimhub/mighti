Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)
=============================================================================

**Warning!** MIGHTI is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is *not yet ready* to be used for real research or policy analysis without careful validation.

MIGHTI is an agent-based modeling framework designed to simulate the dynamics of infectious diseases, non-communicable diseases (NCDs), and their interactions over time. MIGHTI builds on the Starsim architecture.

**Documentation:** See the `MIGHTI Wiki <https://github.com/starsimhub/mighti/wiki>`_ for usage instructions and examples.

Requirements
------------

- Python 3.9–3.13 (CI runs on Python 3.12)
- NumPy >= 2.0 (required by STIsim 1.4.0+; older NumPy may crash on ``argsort(..., stable=True)``)
- Starsim 3.0.x (tested with ``starsim==3.0.3``)
- STIsim 1.4.0 (tested with ``STIsim==1.4.0``)

Installation
------------

MIGHTI is **not yet available on PyPI**, but you can install it directly from GitHub:

.. code-block:: bash

    git clone https://github.com/starsimhub/mighti.git
    cd mighti

For a fully reproducible environment (recommended), install pinned dependencies:

.. code-block:: bash

    pip install -r requirements.txt
    pip install -e .

If you installed with ``pip install -e .`` but are seeing an error like
``TypeError: argsort() got an unexpected keyword argument 'stable'``, your NumPy
is too old. Fix with:

.. code-block:: bash

    pip install -U "numpy>=2.0.0"


Running an Example
------------

.. code-block:: bash

    python mighti_main.py

This will run a sample simulation that includes demography, HIV, and NCD modules.
The example script saves a quick-check plot into the ``outputs/`` folder.


Usage and Documentation
------------

MIGHTI is based on Starsim, please refer to `Starsim documentation <https://docs.idmod.org/projects/starsim/en/latest/>`_ for additional information.

Data policy (raw vs curated)
------------

MIGHTI main is released as a **modeling platform/library** plus small example driver scripts.
To keep releases lightweight and reduce licensing/provenance risk, we follow this convention:

- **Included in releases**:
  - Curated, minimal example inputs in ``data/processed/`` (enough to run the example scripts)
  - Small fixtures in ``tests/test_data/`` used by the automated test suite
- **Not included in stable releases**:
  - ``data/raw/`` and associated cleaning/pre-processing scripts and raw source files

If you need to rebuild curated inputs from raw sources, see ``data/raw/README.md`` and related tooling in this repository (development use).

Public API and Stability
------------

MIGHTI is still evolving. To make it safe for others to build on, we define a **stable public API surface** and treat everything else as internal/experimental.

**Stable entrypoints (preferred usage)**

- Import MIGHTI as a namespace package:

.. code-block:: python

    import mighti as mi

- Use **namespaces** for most functionality (stable module-level entrypoints):

.. code-block:: python

    # Diseases
    t2d = mi.diseases.Type2Diabetes(...)

    # Analyzers
    prev = mi.analyzers.PrevalenceAnalyzer_HIV(...)

    # Connectors / interactions
    conn = mi.interactions.NCDHIVConnector(...)

    # Social determinants of health (SDoH)
    housing = mi.sdoh.NeighbourhoodSituation(...)

    # Interventions
    art = mi.interventions.ARTwithCASM(...)

**Plotting policy**

Plotting utilities are available, but are intentionally **not** imported into the top-level namespace. Import explicitly when needed:

.. code-block:: python

    from mighti.analysis.plotting import plot_mean_prevalence

**Backwards compatibility**

Some older code may still work with `mi.SomeClass` due to a temporary compatibility shim. New code should prefer the namespace style above.

**Internal/experimental (may change without notice)**

- Anything in `mighti.calibration` and most stored calibration artifacts
- Service-use analyzers in `mighti.analyzers.analyzer_serviceuse` (currently stubs)
- Scripts in the repo root (e.g., `mighti_main.py`) are examples/drivers, not API contracts



Contributing
------------

Contributions to the MIGHTI project are welcome! Please read `CONTRIBUTING.rst <https://github.com/starsimhub/mighti/blob/main/CONTRIBUTING.rst>`_ for details on our code of conduct, and the process for submitting pull requests.


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
