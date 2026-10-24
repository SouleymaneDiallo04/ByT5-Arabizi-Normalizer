"""Configuration Sphinx pour la documentation de ByT5-Arabizi-Normalizer."""

project = "ByT5-Arabizi-Normalizer"
author = "Hinimdou Morsia Guitdam, Souleymane Diallo"
copyright = f"2026, {author}"
release = "0.1.0"

language = "fr"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Thème
html_theme = "furo"
html_title = "ByT5-Arabizi-Normalizer"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/SouleymaneDiallo04/ByT5-Arabizi-Normalizer/",
    "source_branch": "main",
    "source_directory": "docs/",
}
