# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib
import sys

# If the package is not installed, point Sphinx at the source tree so autoapi
# can discover the modules without needing an editable install.
sys.path.insert(0, os.path.abspath(".."))


def _read_pyproject_version() -> str:
    """Parse the project version out of ../pyproject.toml.

    We use a tiny stdlib-only parser instead of tomllib (which is 3.11+) so
    that the docs still build on Python 3.10 — the minimum version this
    project supports.
    """
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "0.0.0"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped and "tool.poetry" not in stripped:
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'")
    return "0.0.0"


# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "EMG2MCU"
copyright = "2026, LENeR-UEL"
author = "LENeR-UEL"
# Read from pyproject.toml so the rendered version stays in sync with the
# release workflow's auto-bump.  Can still be overridden on the command line
# via `sphinx-build -D release=X.Y.Z ...`.
release = _read_pyproject_version()
version = release.split(".")[0] + "." + release.split(".")[1]

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    # Auto-generate API reference pages from docstrings — works without importing
    # the package (safer for CI environments that may lack heavy ML deps).
    "autoapi.extension",
    # Google / NumPy docstring support inside napoleon.
    "sphinx.ext.napoleon",
    # Cross-link to NumPy, pandas, scikit-learn, matplotlib docs.
    "sphinx.ext.intersphinx",
    # "View Source" link on every auto-generated page.
    "sphinx.ext.viewcode",
    # Render Markdown files (index, quickstart, etc.) via MyST.
    "myst_parser",
    # Copy-button on all code blocks.
    "sphinx_copybutton",
    # Grid cards, tabs, badges — required for ::::{grid} in index.md.
    "sphinx_design",
]

# Let MyST parse both .md and .rst files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # The docs/README.md is a contributor-facing guide (how to build the
    # docs locally), not part of the rendered site.  Exclude it from
    # Sphinx so it doesn't trigger a "not in any toctree" warning.
    "README.md",
]

# Suppress noisy, non-fatal warnings from auto-generated API pages.
# - "autoapi" silences "Unknown type: placeholder" (sphinx-autoapi/astroid
#   cannot resolve C-extension-backed types from pandas/sklearn).
# - "docutils" silences "Definition list ends without a blank line" and
#   "Block quote ends without a blank line" — both fire on auto-generated
#   RST for docstrings that nest dict/list literals inside a Returns
#   block.  The rendered HTML is unaffected.
# - "toc.not_included" silences the auto-generated api/EMG2MCU/index.rst
#   (its toctree is wired up via api_reference.md, not the package
#   index itself).
suppress_warnings = ["autoapi", "docutils", "toc.not_included"]

# ---------------------------------------------------------------------------
# sphinx-autoapi
# ---------------------------------------------------------------------------
autoapi_dirs = ["../EMG2MCU"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
# Don't re-document members imported from other modules (avoids duplicates
# when e.g. something is re-exported from both a submodule and __init__).
autoapi_keep_files = False
# Put the auto-generated pages under /api/
autoapi_root = "api"
# Don't add autoapi to the TOC automatically — we control placement in index.
autoapi_add_toctree_entry = False
# Render docstrings using napoleon (Google-style).
autoapi_python_use_implicit_namespaces = False

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Intersphinx — cross-link to external project docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ---------------------------------------------------------------------------
# MyST parser options
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# ---------------------------------------------------------------------------
# HTML output — pydata-sphinx-theme (same look as NumPy / pandas / sklearn)
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/LENeR-UEL/EMG2MCU",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_nav_level": 10,
    "navigation_with_keys": False,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/emg2mcu/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "github_user": "LENeR-UEL",
    "github_repo": "EMG2MCU",
    "github_version": "main",
    "doc_path": "docs",
}

html_title = "EMG2MCU"
html_short_title = "EMG2MCU"
html_css_files = ["custom.css"]

# Remove the left sidebar from pages that don't need it.
html_sidebars = {
    "index": [],
    "quickstart": [],
}
