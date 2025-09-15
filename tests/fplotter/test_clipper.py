import pytest
import numpy as np

from fplotter.sampler import Sample2d
import fplotter.clipper as clipper


@pytest.fixture
def standard_sample():
    y = np.array([0, 1, 2, 3, 4, 5, 100])
    x = np.arange(len(y))
    return Sample2d(x, y, len(x))


class TestComputeFocusZone:
    def test_standard_calculation(self):
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        k = 1.5
        # Q1 = 3.25, Q3 = 7.75, IQR = 4.5 -> Lower = -3.5, Upper=14.5
        assert clipper._compute_focus_zone(y, k) is None

    def test_with_nan_values(self):
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan])
        k = 1.5
        assert clipper._compute_focus_zone(y, k) is None

    def test_all_nans_returns_none(self):
        y = np.repeat(np.nan, 10)
        assert clipper._compute_focus_zone(y, 1.5) is None

    def test_no_variance_returns_none(self):
        y = np.array([5, 5, 5, 5, 5])
        assert clipper._compute_focus_zone(y, 1.5) is None


class TestClip:
    def test_with_provided_bound(self, standard_sample):
        bound = (2.5, 10.0)
        # y = [0, 1, 2, 3, 4, 5, 100]
        # Values outside bound: 0, 1, 2, 100
        expected_mask = np.array([True, True, True, False, False, False, True])
        result_mask, _ = clipper.clip(standard_sample, bound=bound, k=1.5)
        assert np.allclose(result_mask, expected_mask)

    def test_with_computed_bound(self, standard_sample):
        # y = [0, 1, 2, 3, 4, 5, 100]
        # Q1 = 1.5, Q3 = 4.5, IQR = 3, k=1.5 -> bound = [-3, 9]
        expected_mask = np.array(
            [False, False, False, False, False, False, True]
        )
        expected_bound = (-3, 9)
        result_mask, result_bound = clipper.clip(standard_sample, None, 1.5)
        assert np.allclose(result_mask, expected_mask)
        assert np.allclose(result_bound, expected_bound)

    def test_returns_all_false_when_bound_is_none(self):
        sample = Sample2d(np.arange(4), np.repeat(5, 4), 4)
        result_mask, result_bound = clipper.clip(sample, None, 1.5)
        expected_mask = np.repeat(False, len(sample.data))
        assert np.allclose(result_mask, expected_mask)
        assert result_bound is None

    def test_returns_all_false_for_all_nan_sample(self):
        sample = Sample2d(np.arange(4), np.repeat(np.nan, 4), 4)
        result_mask, result_bound = clipper.clip(sample, bound=None, k=1.5)
        expected_mask = np.repeat(False, len(sample.data))
        assert np.allclose(result_mask, expected_mask)
        assert result_bound is None
