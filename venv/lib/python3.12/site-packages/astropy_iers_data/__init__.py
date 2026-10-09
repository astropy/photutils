# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pathlib

from ._version import __version__

__all__ = [
    "IERS_A_FILE",
    "IERS_A_URL",
    "IERS_A_URL_MIRROR",
    "IERS_A_README",
    "IERS_B_FILE",
    "IERS_B_URL",
    "IERS_B_README",
    "IERS_LEAP_SECOND_FILE",
    "IERS_LEAP_SECOND_URL",
    "IERS_LEAP_SECOND_URL_MIRROR",
]

DATA = pathlib.Path(__file__).resolve().parent / "data"

# IERS-A default file name, URL, and ReadMe with content description
IERS_A_FILE = str(DATA / "finals2000A.all")
IERS_A_URL = "https://datacenter.iers.org/data/9/finals2000A.all"
IERS_A_URL_MIRROR = "https://maia.usno.navy.mil/ser7/finals2000A.all"
IERS_A_README = str(DATA / "ReadMe.finals2000A")

# IERS-B default file name, URL, and ReadMe with content description
IERS_B_FILE = str(DATA / "eopc04.1962-now")
IERS_B_URL = "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now"
IERS_B_README = str(DATA / "ReadMe.eopc04")

# LEAP SECONDS default file name, URL, and alternative format/URL
IERS_LEAP_SECOND_FILE = str(DATA / "Leap_Second.dat")
IERS_LEAP_SECOND_URL = "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"
IERS_LEAP_SECOND_URL_MIRROR = "https://data.iana.org/time-zones/data/leap-seconds.list"
