# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The tests for the mask_array code. This code tests against the results of aperture
#photometry
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np

from ..aperture_core import *
from ..mask_data import *

apertures = {'edge': CircularAperture([(0, 0)], r = 2.0),
             'no_overlap': CircularAperture([(100, 100)], r = 2.0),
             'full_overlap': CircularAperture([(5, 5)], r = 2.0),
             'fractional_pixel': CircularAperture([(5.5, 5.5)], r = 2.0),
             'different_radius': CircularAperture([(5, 5)], r = 3.0),
             'multiple_objects': CircularAperture([(5, 5), (10, 13)], r = 2.0)}



def insert_source(apertures):
    img_dim = (30, 30)
    img = np.ones(img_dim)
    for aperture in apertures:
        for xcenter, ycenter in aperture.positions:
            if xcenter < img_dim[0] and ycenter < img_dim[1]:
                for xpix in range(int(xcenter) - 3, int(xcenter) + 4):
                    for ypix in range(int(ycenter) - 3, int(ycenter) + 4):
                        img[xpix, ypix] = 5 - max(abs(xcenter - xpix), abs(ycenter - ypix))
    return img

def aperture_test(aperture, img):
    returned_list = get_cutouts(img, aperture)
    phot_table = aperture_photometry(img, aperture)
    test_flux = []
    for indiv_obj in returned_list:
        data, mask = indiv_obj
        test_flux.append(np.sum(data*mask))
    return np.array(test_flux), phot_table['aperture_sum'].data

def test_aperture_edge():
    test_img = insert_source(apertures.values())
    test_flux, phot_flux = aperture_test(apertures['edge'], test_img)
    assert np.allclose(test_flux, phot_flux, rtol = 0.01), "Edge case failed, test flux= {}, phot flux = {}".format(test_flux, phot_flux)

def test_no_overlap():
    test_img = insert_source(apertures.values())
    data, mask =  get_cutouts(test_img, apertures['no_overlap'])[0]
    assert len(data) == 0, 'No overlap case failed, {}'.format(data)
    assert len(mask) == 0, 'No overlap case failed, {}'.format(mask)


def test_basic_case():
    test_img = insert_source(apertures.values())
    test_flux, phot_flux = aperture_test(apertures['full_overlap'], test_img)
    assert np.allclose(test_flux, phot_flux, rtol = 0.01), "Full overlap case failed, test flux= {}, phot flux = {}".format(test_flux, phot_flux)

def test_fractional_pixel_center():
    test_img = insert_source(apertures.values())
    test_flux, phot_flux = aperture_test(apertures['fractional_pixel'], test_img)
    assert np.allclose(test_flux, phot_flux, rtol = 0.01), "fractional pixel case failed, test flux= {}, phot flux = {}".format(test_flux, phot_flux)

def test_diff_radius():
    test_img = insert_source(apertures.values())
    test_flux, phot_flux = aperture_test(apertures['different_radius'], test_img)
    assert np.allclose(test_flux, phot_flux, rtol = 0.01), "different radius case failed, test flux= {}, phot flux = {}".format(test_flux, phot_flux)

def test_multiple_objects():
    test_img = insert_source(apertures.values())
    test_flux, phot_flux = aperture_test(apertures['multiple_objects'], test_img)
    assert np.allclose(test_flux, phot_flux, rtol = 0.01), "multiple objects case failed, test flux= {}, phot flux = {}".format(test_flux, phot_flux)
