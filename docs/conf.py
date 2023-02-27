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
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery'
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
master_doc = 'index'
bibtex_bibfiles = ['ref.bib']
source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_favicon = 'fairpy_ico.png'





# # -- Options for HTMLHelp output ---------------------------------------------

# # Output file base name for HTML help builder.
# htmlhelp_basename = 'fairpydoc'

# # -- Options for LaTeX output ------------------------------------------------

# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     #
#     # 'papersize': 'letterpaper',

#     # The font size ('10pt', '11pt' or '12pt').
#     #
#     # 'pointsize': '10pt',

#     # Additional stuff for the LaTeX preamble.
#     #
#     # 'preamble': '',

#     # Latex figure (float) alignment
#     #
#     # 'figure_align': 'htbp',
# }

# # Grouping the document tree_ into LaTeX files. List of tuples
# # (source start file, target name, title,
# #  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     (master_doc, 'fairpy.tex', 'FairPy Documentation',
#      'FairPy Team', 'manual'),
# ]

# # -- Options for manual page output ------------------------------------------

# # One entry per manual page. List of tuples
# # (source start file, name, description, authors, manual section).
# man_pages = [
#     (master_doc, 'fairpy', 'FairPy Documentation',
#      [author], 1)
# ]

# # -- Options for Texinfo output ----------------------------------------------

# # Grouping the document tree_ into Texinfo files. List of tuples
# # (source start file, target name, title, author,
# #  dir menu entry, description, category)
# texinfo_documents = [
#     (master_doc, 'fairpy', 'FairPy Documentation',
#      author, 'FairPy', 'One line description of project.',
#      'Miscellaneous'),
# ]