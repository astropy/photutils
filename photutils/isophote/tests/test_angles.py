# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the angles module.
"""

import math

import numpy as np


def sector_area(a, eps, phi, r):
    aux = r * np.cos(phi) / a
    saux = aux / abs(aux)
    if abs(aux) >= 1.0:
        aux = saux
    return abs(a**2 * (1.0 - eps) / 2.0 * math.acos(aux))


def test_angles(phi_min=0.05, phi_max=0.2):
    a = 40.0
    astep = 1.1
    eps = 0.1

    # r = a
    a1 = a * (1.0 - ((1.0 - 1.0 / astep) / 2.0))
    a2 = a * (1.0 + (astep - 1.0) / 2.0)
    r3 = a2
    r4 = a1
    aux = min((a2 - a1), 3.0)
    sarea = (a2 - a1) * aux
    dphi = max(min((aux / a), phi_max), phi_min)
    phi = dphi / 2.0
    phi2 = phi - dphi / 2.0
    aux = 1.0 - eps
    r3 = a2 * aux / np.sqrt((aux * np.cos(phi2))**2 + (np.sin(phi2))**2)
    r4 = a1 * aux / np.sqrt((aux * np.cos(phi2))**2 + (np.sin(phi2))**2)

    ncount = 0
    while phi < np.pi * 2:
        phi1 = phi2
        r1 = r4
        r2 = r3
        phi2 = phi + dphi / 2.0
        aux = 1.0 - eps
        r3 = a2 * aux / np.sqrt((aux * np.cos(phi2))**2 + (np.sin(phi2))**2)
        r4 = a1 * aux / np.sqrt((aux * np.cos(phi2))**2 + (np.sin(phi2))**2)

        sa1 = sector_area(a1, eps, phi1, r1)
        sa2 = sector_area(a2, eps, phi1, r2)
        sa3 = sector_area(a2, eps, phi2, r3)
        sa4 = sector_area(a1, eps, phi2, r4)
        area = abs((sa3 - sa2) - (sa4 - sa1))

        # Compute step to next sector and its angular span
        dphi = max(min((sarea / (r3 - r4) / r4), phi_max), phi_min)
        phistep = dphi / 2.0 + phi2 - phi

        ncount += 1

        assert 11.0 < area < 12.4

        phi = phi + min(phistep, 0.5)
        # r = (a * (1.0 - eps) / np.sqrt(((1.0 - eps) * np.cos(phi))**2 +
        #                               (np.sin(phi))**2))

    assert ncount == 72
