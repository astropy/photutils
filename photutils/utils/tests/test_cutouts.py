# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..cutouts import cutout_footprint


XCS = [25.7]
YCS = [26.2]
XSTDDEVS = [3.2, 4.0]
YSTDDEVS = [5.7, 4.1]
THETAS = np.array([30., 45.]) * np.pi / 180.
DATA = np.zeros((3, 3))
DATA[0:2, 1] = 1.
DATA[1, 0:2] = 1.
DATA[1, 1] = 2.


class TestCutoutFootprint:
    def test_dataonly(self):
        data = np.ones((5, 5))
        position = (2, 2)
        result1 = cutout_footprint(data, position, 3)
        result2 = cutout_footprint(data, position, footprint=np.ones((3, 3)))
        assert_allclose(result1[:-2], result2[:-2])
        assert result1[-2] is None
        assert result2[-2] is None
        assert result1[-1] == result2[-1]

    def test_mask_error(self):
        data = error = np.ones((5, 5))
        mask = np.zeros_like(data, dtype=bool)
        position = (2, 2)
        box_size1 = 3
        box_size2 = (3, 3)
        footprint = np.ones((3, 3))
        result1 = cutout_footprint(data, position, box_size1, mask=mask,
                                   error=error)
        result2 = cutout_footprint(data, position, box_size2, mask=mask,
                                   error=error)
        result3 = cutout_footprint(data, position, box_size1,
                                   footprint=footprint, mask=mask,
                                   error=error)
        assert_allclose(result1[:-1], result2[:-1])
        assert_allclose(result1[:-1], result3[:-1])
        assert result1[-1] == result2[-1]

    def test_position_len(self):
        with pytest.raises(ValueError):
            cutout_footprint(np.ones((3, 3)), [1])

    def test_nofootprint(self):
        with pytest.raises(ValueError):
            cutout_footprint(np.ones((3, 3)), (1, 1), box_size=None,
                             footprint=None)

    def test_wrongboxsize(self):
        with pytest.raises(ValueError):
            cutout_footprint(np.ones((3, 3)), (1, 1), box_size=(1, 2, 3))
