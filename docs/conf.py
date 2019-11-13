import os
import sys

from sphinx.highlighting import lexers

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

import scvae
from custom_lexers import TerminalLexer

# Project information

project = scvae.__title__
copyright = scvae.__copyright__
author = scvae.__author__
version = scvae.__version__
release = scvae.__version__

# General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon"
]
templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext"
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
default_role = "py:obj"

# autodoc options

autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "loompy",
    "matplotlib",
    "mpl_toolkits",
    "pandas",
    "PIL",
    "sklearn",
    "scipy",
    "seaborn",
    "tables",
    "tensorflow",
    "tensorflow_probability"
]

# HTML options

html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "github_user": "scvae",
    "github_repo": "scvae",
    "github_type": "star",
    "description": "Single-cell variational auto-encoders",
    "show_powered_by": False
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
    ]
}

lexers["terminal"] = TerminalLexer(startinline=True)
