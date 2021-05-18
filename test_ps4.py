import pylab
import re
import pytest
import numpy as np
from numpy import array
from numpy.testing import assert_array_equal

from ps4 import *
from assertpy import assert_that


class TestGenerateModels:
    @pytest.fixture(autouse=True)
    def prepare_generate_models(self, x, y, degs):
        self.generate_models = generate_models(x, y, degs)

    # TEST generate_models()

    # given
    correct_output1 = [array([3.51667102e-02, -1.37199002e+02, 1.33764160e+05])]
    # correct_output2 = [array([  1.10000000e+00,  -2.15270000e+03]),
    #                    array([ -8.86320608e-14,   1.10000000e+00,  -2.15270000e+03])]
    # correct_output3 = [array([  7.64961915e-02,  -1.53745049e+02]),
    #                    array([  9.24663056e-02,  -3.66029000e+02,   3.62195201e+05]),
    #                    array([  7.59680860e-02,  -4.52530612e+02,   8.98514989e+05, -5.94652222e+08])]
    # correct_output4 = [array([  1.43651226e-01,  -2.84888692e+02]),
    #                    array([  1.45050481e-02,  -5.72778422e+01,   5.65389304e+04]),
    #                    array([  2.41178585e-02,  -1.43636842e+02,   2.85137445e+05, -1.88670385e+08]),
    #                    array([  1.14194270e-02,  -9.08068327e+01,   2.70778566e+05, -3.58853930e+08,   1.78337425e+11])]
    given1 = [1900, 1901, 1902, 1904, 2000], [32.0, 42.0, 31.3, 22.0, 33.0], [2], correct_output1

    # given2 = [1961, 1962, 1963], [4.4, 5.5, 6.6], [1, 2], correct_output2
    # given3 = [1960, 1997, 1999, 2001], [-3.1, -4.1, -9.2, 10.1], [1, 2, 3], correct_output3
    # given4 = [1960, 1997, 1999, 2001, 1998, 1995], [-3.1, -4.1, -9.2, 10.1, 9.1, 4.5], [1, 2, 3, 4], correct_output4
    @pytest.mark.parametrize('x, y, degs, expected', [given1])
    def test_generate_models(self, expected):
        # when
        result = self.generate_models
        print(result)
        # then
        assert_array_equal(result, expected)
