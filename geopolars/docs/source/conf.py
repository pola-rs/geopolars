# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Based on the one of py-polars.
# `make docs` runs this from a copy of the source with the `@doc` docstrings filled in.

from __future__ import annotations

import sys
from pathlib import Path

import sphinx_autosummary_accessors

# -- Path setup --------------------------------------------------------------

# Add the geopolars package directory
sys.path.insert(0, str(Path(__file__).parents[2].resolve()))


# -- Project information -----------------------------------------------------

project = "GeoPolars"
author = "GeoPolars contributors"
copyright = f"2026, {author}"


# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # Third-party extensions
    "numpydoc",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
    "sphinx_design",
]

# Render docstring text in `single backticks` as code.
default_role = "code"

maximum_signature_line_length = 88

# sphinx-autosummary-accessors - build docs for namespace accessors like `Expr.geo`
# https://sphinx-autosummary-accessors.readthedocs.io/en/stable/
templates_path = [sphinx_autosummary_accessors.templates_path]

exclude_patterns = ["Thumbs.db", ".DS_Store"]


# -- Extension settings ------------------------------------------------------

# sphinx.ext.intersphinx - link to other projects' documentation
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
}

# numpydoc - parse numpy docstrings
# https://numpydoc.readthedocs.io/en/latest/
numpydoc_show_class_members = False

# Sphinx-copybutton - add copy button to code blocks
# https://sphinx-copybutton.readthedocs.io/en/latest/index.html
# strip the '>>>' and '...' prompt/continuation prefixes.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/geopolars/geopolars",
            "icon": "fa-brands fa-github",
        },
    ],
}
