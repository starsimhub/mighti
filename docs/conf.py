# Configuration file for the Sphinx documentation builder

import os
import sys
import datetime
import runpy

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath('..'))  # Add the project root to sys.path

# -- Project information ------------------------------------------------------
project = 'MIGHTI'
author = 'Your Name or Team Name'
copyright = f'{datetime.datetime.now().year}, {author}'
versionpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mighti', 'util', 'version.py'))
release = runpy.run_path(versionpath)['__version__']

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',     # For Google-style docstrings
    'sphinx.ext.viewcode',     # Adds links to source code
    'numpydoc',                # For NumPy-style docstrings
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output --------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc options ----------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autoclass_content = 'class'  # Include class docstring and __init__

# Optional: suppress warnings from numpydoc
numpydoc_show_class_members = False
