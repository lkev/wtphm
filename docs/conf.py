# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme
# import m2r
# import recommonmark
# from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../wtphm'))
sys.path.insert(0, os.path.abspath('../wtphm/classification'))
sys.path.insert(0, os.path.abspath('../wtphm/clustering'))

# -- Project information -----------------------------------------------------

project = 'wtphm'
copyright = '2020, Kevin Leahy'
author = 'Kevin Leahy'
version = '0.1.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx_rtd_theme'
              # 'recommonmark'
              # 'm2r'
              ]
autodoc_member_order = 'bysource'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# for matplotlib
plot_include_source = False
plot_html_show_source_link = False

napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_param = False

# letting readthedocs know that the master_file is not called contents, it's
# called index
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
                      'collapse_navigation': False,
                      'navigation_depth': -1
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# The below are the main requirements of the project. They are not actually
# needed to build the docs so are saved here as fake packages.
autodoc_mock_imports = [
    "matplotlib", "numpy", "sklearn", "pandas", "operator", "warning",
    "scipy", "math", "itertools"]


# The below is for recommonmark - for converting markdown to rst files
# def setup(app):
#     app.add_config_value('recommonmark_config', {
#             # 'url_resolver': lambda url: github_doc_root + url,
#             # 'auto_toc_tree_section': 'Contents',
#             }, True)
#     app.add_transform(AutoStructify)
