# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module extends the package setup parameters to include data files.
"""


def get_package_data():
    """Include additional data files."""
    return {'photutils.tests': ['coveragerc']}
