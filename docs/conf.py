import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "CosmoPyro"
copyright = "2025, Konstantin Leyde"
author = "Konstantin Leyde"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
# ../images is the single source of truth for the logos, shared with the README.
# Everything listed here is copied into _static/ in the build output, so the
# theme options below still refer to the files by bare name.
html_static_path = ["_static", "../images"]
html_css_files = ["custom.css"]
html_title = ""

# Logo: light/dark variants, shown in the sidebar on every page. Furo resolves
# these relative to _static, so a remote URL would not work here.
html_theme_options = {
    "light_logo": "CosmoPyro_light.png",
    "dark_logo": "CosmoPyro_dark.png",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# --- Generate model_registry.js for the interactive kwargs builder ---
import json

sys.path.insert(0, os.path.dirname(__file__))
from model_registry import COSMO_MODELS, MASS_MODELS, MASS_RATIO_MODELS

# Write model registry as a JS file that works with file:// (no fetch needed)
_js_path = os.path.join(os.path.dirname(__file__), "_static", "model_registry.js")
os.makedirs(os.path.dirname(_js_path), exist_ok=True)
with open(_js_path, "w") as f:
    f.write("// Auto-generated from docs/model_registry.py — do not edit\n")
    f.write("var MODEL_REGISTRY = ")
    json.dump(MASS_MODELS, f, indent=2)
    f.write(";\n")
    f.write("var COSMO_MODEL_REGISTRY = ")
    json.dump(COSMO_MODELS, f, indent=2)
    f.write(";\n")
    f.write("var MASS_RATIO_MODEL_REGISTRY = ")
    json.dump(MASS_RATIO_MODELS, f, indent=2)
    f.write(";\n")
