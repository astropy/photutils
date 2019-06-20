# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module extends the package setup parameters to include the
subpackage data files.
"""


def get_package_data():
    """Include the subpackage data files."""
    return {'photutils.datasets': ['data/*']}
