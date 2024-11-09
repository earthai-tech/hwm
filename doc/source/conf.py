# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Add the hwm package to the Python path
sys.path.insert(0, os.path.abspath('../../'))

import hwm

# -- Path setup --------------------------------------------------------------

# import os
# import sys
# import subprocess

# # # Attempt to add hwm to Python path
# # sys.path.insert(0, os.path.abspath('../../'))

# # # Try importing hwm; if unsuccessful, install it in editable mode
# try:
#     import hwm
# except ImportError:
#     print("hwm not found, attempting installation in editable mode.")
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "../../"])
#     import hwm  # Retry the import after installation


# -- Project information -----------------------------------------------------

project = 'HWM: Adaptive Hammerstein-Wiener'
copyright = '2024, Kouadio Laurent'
author = 'Kouadio Laurent'
release = hwm.__version__

# -- General configuration ---------------------------------------------------

# Sphinx extension modules
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
]

# Templates path
templates_path = ['_templates']

# Patterns to exclude from source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# Using Read the Docs theme for a dynamic and modern look
# import sphinx_wagtail_theme
# html_theme = 'sphinx_pdj_theme'
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
# extensions.append("sphinx_wagtail_theme")
# html_theme = 'sphinx_wagtail_theme'
html_theme='traditional'
html_theme_options = {
    # "logo": "path/to/logo.png",  # Add your own logo file path
    "title": "HWM Documentation",  # Custom title for your project
    #"header_links": [("Home", "/"), ("Docs", "/docs/"), ("API", "/api/")],  # Update links as needed
    "show_search_button": True,
    "show_nav": True,
    "github_url": "https://github.com/earthai-tech/hwm",
    "repository_url": "https://github.com/earthai-tech/hwm",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
}

#html_theme = "alabaster" #sphinx_book_theme" #'sphinx_rtd_theme'

# Static files path (such as style sheets)
html_static_path = ['_static']

# Additional theme options for styling
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'style_nav_header_background': '#2980B9',
}

#Custom sidebar templates
html_sidebars = {
    '**': [
        # 'about.html',
#         'navigation.html',
        'searchbox.html',
    ]
}

# -- Autodoc options ---------------------------------------------------------

# Autodoc settings to automatically document classes, functions, and modules
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Extension settings ------------------------------------------------------

# Intersphinx mapping to Python standard library
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- TODO settings -----------------------------------------------------------
todo_include_todos = True

html_static_path = ["_static"]
html_css_files = ["custom.css"]


