# Model of Inter-Generational Health, Transmission, and Interventions (MIGHTI)

**Warning!** MIGHTI is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is *not yet ready* to be used for real research or policy analysis without careful validation.

MIGHTI is an agent-based modeling framework designed to simulate the dynamics of infectious diseases, non-communicable diseases (NCDs), and their interactions over time. MIGHTI builds on the Starsim architecture.

**Documentation:** See the [MIGHTI Wiki](https://github.com/starsimhub/mighti/wiki) for usage instructions and examples.

---

## Requirements

- Python 3.9–3.13
- Operating System: Windows, macOS, or Linux
- Conda or virtual environment (recommended)

### Required Dependencies

You can install them via `docs/requirements.txt` or manually:

#### Core packages:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`

#### Simulation Engine:
- [`starsim`](https://github.com/starsimhub/starsim) — agent-based simulation core
- [`stisim`](https://github.com/starsimhub/stisim) — sexually transmitted infection modules
- `sciris` — support library for simulation management

#### Calibration (optional):
- `optuna`
- `tqdm`

#### Development tools (optional):
- `pytest` — for testing
- `sphinx`, `myst-parser`, `sphinx_rtd_theme` — for documentation

#### Notebook & IDE support (optional):
- `ipykernel`
- `spyder-kernels==2.5.*`

---

## Installation

MIGHTI is **not yet available on PyPI**, but you can install it directly from GitHub:

```bash
git clone https://github.com/starsimhub/mighti.git
cd mighti
pip install -e .
```

Please also make sure to install its dependencies:

```bash
pip install -r docs/requirements.txt
```

To ensure you're using the correct versions of Starsim and STIsim:

```bash
pip install --upgrade --force-reinstall git+https://github.com/starsimhub/starsim.git
pip install --upgrade --force-reinstall git+https://github.com/starsimhub/stisim.git
```

---

## Running an Example

```bash
python mighti_main.py
```

This will run a sample simulation that includes demography, HIV, and NCD modules. Outputs will be saved in the `outputs/` folder.

You can also run:

- `mighti_demography.py` — mortality and life expectancy module
- `mighti_calibration.py` — Optuna-based parameter calibration

---

## Usage and Documentation

MIGHTI builds on [Starsim](https://docs.idmod.org/projects/starsim/en/latest/). Please refer to the Starsim documentation for details on base classes, intervention logic, and simulation controls.

---

## Contributing

Contributions to the MIGHTI project are welcome!  
Please read the [CONTRIBUTING.rst](https://github.com/starsimhub/mighti/blob/main/contributing.rst) for guidelines on submitting pull requests and our code of conduct.

---

## Data Sources

MIGHTI uses public datasets for demographic and disease burden modeling:

### Demography
- [UN World Population Prospects 2024](https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality)

### Disease Burden
- [Global Burden of Disease Study (GBD)](https://vizhub.healthdata.org/gbd-results/)

### HIV
- [Demographic and Health Surveys (DHS)](https://dhsprogram.com/pubs/pdf/FR202/FR202.pdf)
- [Swaziland HIV Incidence Measurement Survey (SHIMS)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5291824/)

---

## Disclaimer

This code was developed by researchers at NYU, IDM, and collaborators.  
It is shared under the MIT License to foster reproducibility and future development.  
No guarantees are made regarding functionality or support.  
You are free to fork and modify the code under the terms of the license.
