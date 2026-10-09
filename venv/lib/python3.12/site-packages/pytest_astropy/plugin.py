# Licensed under a 3-clause BSD style license - see LICENSE.rst

def pytest_addoption(parser):
    parser.addoption(
        "-R", nargs="?", const='any', default='none',
        help="run tests with online data, requires pytest-remotedata",
        dest="remote_data", choices=['astropy', 'any', 'github', 'none'])
