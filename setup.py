import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'mighti', 'util', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.rst'), "r") as f:
    long_description = f.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

setup(
    name="mighti",
    version=version,
    author="Authorship TBC",
    license="MIT",
    description="MIGHTI: Model of Inter-Generational Health, Transmission, and Interventions",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=2.0.0',
        'starsim>=3.0.3',
        'stisim>=1.4.0',
        'scipy',
        'pandas>=2.0.0',
        'sciris>=3.0.0',
        'matplotlib',
        'seaborn',
    ],
)
