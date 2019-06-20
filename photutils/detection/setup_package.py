# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module extends the package setup parameters to include the
subpackage test data files.
"""


def get_package_data():
    """Include the subpackage test data files."""
    return {'photutils.detection.tests': ['data/*.txt']}
