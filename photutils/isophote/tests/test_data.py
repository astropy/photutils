#
# Pointer to test data
#
# For this to work, download the astropy/photutils-datasets package
# and install it at the same level in the directories tree where
# the photutils package is stored.
#
from os import path

_TEST_SCRIPT_PATH = path.dirname(path.dirname(path.dirname(__file__)))

TEST_DATA = _TEST_SCRIPT_PATH + "/../../photutils-datasets/data/isophote/"
TEST_DATA_REGRESSION = _TEST_SCRIPT_PATH + "/isophote/tests/data/"

