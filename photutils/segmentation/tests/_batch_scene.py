# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared synthetic scene for the batch-driver comparator tests.
"""

import numpy as np

from photutils.segmentation import SourceCatalog, detect_sources

__all__ = ['make_batch_scene']


def make_batch_scene(*, seed=0):
    """
    Make a deterministic test scene exercising the batch-driver edge
    cases.

    The scene contains isolated sources, close/overlapping pairs
    (neighbor-segment handling), sources touching every image edge (bbox
    clipping), masked pixels inside sources and spanning a close pair
    (mirror-correction rejection), and non-finite data values inside a
    source.

    Parameters
    ----------
    seed : int, optional
        The seed for the random number generator used for the
        background noise.

    Returns
    -------
    result : dict
        Keys ``data``, ``error``, ``mask``, ``segm`` (a
        `~photutils.segmentation.SegmentationImage`).
    """
    rng = np.random.default_rng(seed)
    ny = nx = 151
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = rng.normal(0.0, 0.1, (ny, nx))
    positions = [(20, 20), (24, 26), (75, 75), (75, 81), (140, 20),
                 (5, 100), (100, 4), (147, 147), (50, 120), (120, 50),
                 (100, 100)]
    for i, (xc, yc) in enumerate(positions):
        amp = 5.0 + i
        sig = 1.5 + 0.2 * (i % 4)
        data += amp * np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2)
                             / (2 * sig ** 2))
    error = np.full((ny, nx), 0.1)
    error[::17, ::13] = 0.3
    mask = np.zeros((ny, nx), dtype=bool)
    mask[18:20, 22:24] = True  # inside a source of a close pair
    mask[73:75, 72:84] = True  # spans a close pair
    data[77, 78] = np.nan  # non-finite inside a source
    data[75, 83] = np.inf
    segm = detect_sources(data, 1.0, n_pixels=5)
    return {'data': data, 'error': error, 'mask': mask, 'segm': segm}


def make_catalog(scene, *, aperture_mask_method='correct',
                 with_error=True, with_mask=True):
    """
    Make a `SourceCatalog` from a scene with the given options.

    Parameters
    ----------
    scene : dict
        A scene from `make_batch_scene`.

    aperture_mask_method : {'correct', 'mask', 'none'}, optional
        The segmentation masking method.

    with_error : bool, optional
        Whether to pass the scene error array to the catalog.

    with_mask : bool, optional
        Whether to pass the scene mask array to the catalog.

    Returns
    -------
    result : `~photutils.segmentation.SourceCatalog`
        The source catalog.
    """
    return SourceCatalog(
        scene['data'], scene['segm'],
        error=scene['error'] if with_error else None,
        mask=scene['mask'] if with_mask else None,
        aperture_mask_method=aperture_mask_method)
