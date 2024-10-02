# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Documentation build configuration file.

This file is execfile()d with the current directory set to its
containing dir.

Note that not all possible configuration values are present in this
file.

All configuration values have a default. Some values are defined in the
global Astropy configuration which is loaded here before anything else.
See astropy.sphinx.conf for which values are set there.
"""

import os
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from sphinx.util import logging

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

logger = logging.getLogger(__name__)

try:
    from sphinx_astropy.conf.v2 import *  # noqa: F403
    from sphinx_astropy.conf.v2 import extensions  # noqa: E402
except ImportError:
    msg = ('The documentation requires the sphinx-astropy package to be '
           'installed. Please install the "docs" requirements.')
    logger.error(msg)
    sys.exit(1)

# Get configuration information from pyproject.toml
with (Path(__file__).parents[1] / 'pyproject.toml').open('rb') as fh:
    project_meta = tomllib.load(fh)['project']

# -- Plot configuration -------------------------------------------------------
plot_rcparams = {
    'axes.labelsize': 'large',
    'figure.figsize': (6, 6),
    'figure.subplot.hspace': 0.5,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'none',
}
plot_apply_rcparams = True
plot_html_show_source_link = True
plot_formats = ['png', 'hires.png', 'pdf', 'svg']
# Don't use the default - which includes a numpy and matplotlib import
plot_pre_code = ''

# -- General configuration ----------------------------------------------------
# By default, highlight as Python 3.
highlight_language = 'python3'

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.0'

# Extend astropy intersphinx_mapping with packages we use here
intersphinx_mapping.update(  # noqa: F405
    {'regions': ('https://astropy-regions.readthedocs.io/en/stable/', None),
     'skimage': ('https://scikit-image.org/docs/stable/', None),
     'gwcs': ('https://gwcs.readthedocs.io/en/latest/', None)})

# Exclude astropy intersphinx_mapping for unused packages
del intersphinx_mapping['h5py']  # noqa: F405

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# .inc.rst mean *include* files, don't have sphinx process them
# exclude_patterns += ["_templates", "_pkgtemplate.rst", "**/*.inc.rst"]

extensions += [
    'sphinx_design',
]

# This is added to the end of RST files - a good place to put
# substitutions to be used globally.
rst_epilog = """
.. _Astropy: https://www.astropy.org/
"""

# -- Project information ------------------------------------------------------
project = project_meta['name']
author = project_meta['authors'][0]['name']
project_copyright = f'2011-{datetime.now(tz=timezone.utc).year}, {author}'
github_project = 'astropy/photutils'

# The version info for the project you're documenting, acts as
# replacement for |version| and |release|, also used in various other
# places throughout the built documents.

# The full version, including alpha/beta/rc tags.
release = metadata.version(project)
# The short X.Y version.
version = '.'.join(release.split('.')[:2])
dev = 'dev' in release

# -- Options for HTML output --------------------------------------------------

html_theme_options.update(  # noqa: F405
    {
        'header_links_before_dropdown': 6,
        'github_url': 'https://github.com/astropy/photutils',
        'use_edit_page_button': False,
        'logo': {
            'image_light': 'photutils_banner.svg',
            'image_dark': 'photutils_banner.svg',
        },
        'navigation_with_keys': False,
        'content_footer_items': ['last-updated'],
    }
)

html_show_sourcelink = False

# The global astropy configuration uses a custom theme,
# 'bootstrap-astropy', which is installed along with astropy. A
# different theme can be used or the options for this theme can be
# modified by overriding some of the variables set in the global
# configuration. The variables set in the global configuration are
# listed below, commented out.

# Add any paths that contain custom themes here, relative to this
# directory.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages. See the documentation
# for a list of builtin themes. To override the custom theme, set this
# to the name of a builtin theme or the name of a custom theme in
# html_theme_path.
# html_theme = None

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# The name of an image file (relative to this directory) to place at the
# top of the sidebar.
# html_logo = ''

# The name of an image file (within the static path) to use as favicon
# of the docs. This file should be a Windows icon file (.ico) being
# 16x16 or 32x32 pixels large.
html_favicon = os.path.join('_static', 'favicon.ico')

# A "Last built" timestamp is inserted at every page bottom, using the
# given strftime format. Set to '' to omit this timestamp.
# html_last_updated_fmt = '%d %b %Y'

# The name for this set of Sphinx documents. If None, it defaults to
# "<project> v<release>".
html_title = f'{project} {release}'

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# Static files to copy after template files
html_static_path = ['_static']
html_style = 'photutils.css'

# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get('READTHEDOCS_CANONICAL_URL', '')

# A dictionary of values to pass into the template engine's context for
# all pages.
html_context = {
    'default_mode': 'light',
    'to_be_indexed': ['stable', 'latest'],
    'is_development': dev,
    'github_user': 'astropy',
    'github_repo': 'photutils',
    'github_version': 'main',
    'doc_path': 'docs',
    # Tell Jinja2 templates the build is running on Read the Docs
    'READTHEDOCS': os.environ.get('READTHEDOCS', '') == 'True',
}

# -- Options for LaTeX output -------------------------------------------------
# Grouping the document tree into LaTeX files. List of tuples (source
# start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + ' Documentation',
                    author, 'manual')]
latex_logo = '_static/photutils_banner.pdf'

# -- Options for manual page output -------------------------------------------
# One entry per manual page. List of tuples (source start file, name,
# description, authors, manual section).
man_pages = [('index', project.lower(), project + ' Documentation',
              [author], 1)]

# -- Resolving issue number to links in changelog -----------------------------
github_issues_url = f'https://github.com/{github_project}/issues/'

# -- Turn on nitpicky mode for sphinx (to warn about references not found) ----
nitpicky = True
nitpick_ignore = [('py:obj', 'Model')]

# Some warnings are impossible to suppress, and you can list specific
# references that should be ignored in a nitpick-exceptions file which
# should be inside the docs/ directory. The format of the file should be:
#
# <type> <class>
#
# for example:
#
# py:class astropy.io.votable.tree.Element
# py:class astropy.io.votable.tree.SimpleElement
# py:class astropy.io.votable.tree.SimpleElementWithContent
#
# Uncomment the following lines to enable the exceptions:
nitpick_filename = 'nitpick-exceptions.txt'
if os.path.isfile(nitpick_filename):
    with open(nitpick_filename) as fh:
        for line in fh:
            if line.strip() == '' or line.startswith('#'):
                continue
            dtype, target = line.split(None, 1)
            target = target.strip()
            nitpick_ignore.append((dtype, target))

# -- Options for linkcheck output ---------------------------------------------
linkcheck_retry = 5
linkcheck_ignore = ['http://data.astropy.org',
                    r'https://iraf.net/*',
                    r'https://github\.com/astropy/photutils/(?:issues|pull)/\d+']
linkcheck_timeout = 180
