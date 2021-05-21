# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Documentation build configuration file.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything
# else. See astropy.sphinx.conf for which values are set there.

from configparser import ConfigParser
from datetime import datetime
import os
import sys

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to '
          'be installed')
    sys.exit(1)

# Get configuration information from setup.cfg
conf = ConfigParser()
conf.read([os.path.join(os.path.dirname(__file__), '..', 'setup.cfg')])
setup_cfg = dict(conf.items('metadata'))


# -- General configuration ----------------------------------------------------
# By default, highlight as Python 3.
highlight_language = 'python3'

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.7'

# Extend astropy intersphinx_mapping with packages we use here
intersphinx_mapping['skimage'] = ('https://scikit-image.org/docs/stable/', None)  # noqa
intersphinx_mapping['gwcs'] = ('https://gwcs.readthedocs.io/en/latest/', None)  # noqa

# Exclude astropy intersphinx_mapping for unused packages
del intersphinx_mapping['h5py']  # noqa

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append('_templates')  # noqa
# Exclude template PSF block specification documentation
exclude_patterns.append('psf_spec/*')  # noqa

plot_formats = ['png', 'hires.png', 'pdf', 'svg']

# This is added to the end of RST files - a good place to put
# substitutions to be used globally.
rst_epilog = """
.. _Astropy: https://www.astropy.org/
"""


# -- Project information ------------------------------------------------------
project = setup_cfg['name']
author = setup_cfg['author']
copyright = f'2011-{datetime.utcnow().year}, {author}'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
__import__(project)
package = sys.modules[project]

# The short X.Y version.
version = package.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__


# -- Options for HTML output --------------------------------------------------
# The global astropy configuration uses a custom theme,
# 'bootstrap-astropy', which is installed along with astropy. A different
# theme can be used or the options for this theme can be modified by
# overriding some of the variables set in the global configuration. The
# variables set in the global configuration are listed below, commented
# out.

# Add any paths that contain custom themes here, relative to this
# directory.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages. See the documentation
# for a list of builtin themes. To override the custom theme, set this
# to the name of a builtin theme or the name of a custom theme in
# html_theme_path.
# html_theme = None

# Customized theme options
html_theme_options = {
    'logotext1': 'phot',  # white, semi-bold
    'logotext2': 'utils',  # orange, light
    'logotext3': ''  # white, light
}

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# The name of an image file (relative to this directory) to place at the
# top of the sidebar.
# html_logo = ''

# The name of an image file (within the static path) to use as favicon
# of the docs. This file should be a Windows icon file (.ico) being 16x16
# or 32x32 pixels large.
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


# -- Options for LaTeX output -------------------------------------------------
# Grouping the document tree into LaTeX files. List of tuples (source
# start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]
latex_logo = '_static/photutils_banner.pdf'


# -- Options for manual page output -------------------------------------------
# One entry per manual page. List of tuples (source start file, name,
# description, authors, manual section).
man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]


# -- Resolving issue number to links in changelog -----------------------------
github_project = setup_cfg['github_project']
github_issues_url = f'https://github.com/{github_project}/issues/'


# -- Turn on nitpicky mode for sphinx (to warn about references not found) ----
nitpicky = True
nitpick_ignore = []

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
    for line in open(nitpick_filename):
        if line.strip() == "" or line.startswith("#"):
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
linkcheck_anchors = False
