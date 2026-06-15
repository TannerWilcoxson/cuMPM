import os
import sys
# Use the installed cuMPM package from the environment rather than the raw src directory,
# to ensure the compiled C++ binary extension _cuMPM is correctly loaded.


project = 'cuMPM'
copyright = '2026, Tanner'
author = 'Tanner'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_logo = '_static/cumpm_logo.png'
html_static_path = ['_static']

# Intersphinx mapping to NumPy and Python docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Autodoc settings
autodoc_member_order = 'bysource'
autoclass_content = 'both'
