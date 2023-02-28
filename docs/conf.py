# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'FairPy'
copyright = '2023, FairPy Team'
author = 'FairPy Team'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'fairpy',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


# The master toctree document.
bibtex_bibfiles = ['ref.bib']

# # List of patterns, relative to source directory, that match files and
# # directories to ignore when looking for source files.
# # This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_favicon = 'fairpy_ico.png'
html_logo = 'fairpy_ico.png'