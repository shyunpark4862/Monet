import numpy as np
import pytest

from fplotter.sampler import Sample2d, Sample3d, Sample
from fplotter.resampler import resample, _sort


def univariate_func(x):
    return np.sin(x)


def bivariate_func(x, y):
    return np.sin(x) * y + np.cos(y) * x


@pytest.fixture
def univariate_data():
    x = np.array([3.000000002, 0, 1, 1.9999999999, 4, 2, 3, 1.000000001])
    y = univariate_func(x)
    return Sample2d(x, y, None)


@pytest.fixture
def bivariate_data():
    x = np.array([0, 1, 0, 1, 0.5])
    y = np.array([0, 0, 1, 1, 0.5])
    z = bivariate_func(x, y)
    return Sample3d(x, y, z, None)


def test_sort():
    arr = np.array([5, 1, 4, 1.000001, 3, 5.000002])
    expected = np.array([1, 3, 4, 5])
    result = _sort(arr, 1e-3)
    assert np.allclose(result, expected)


def test_resample_univariate(univariate_data):
    result = resample(univariate_data)
    assert result.n_samples == 5

    expected_x = np.array([0, 1, 2, 3, 4])
    expected_y = univariate_func(expected_x)
    assert np.allclose(result.data[:, 0], expected_x)
    assert np.allclose(result.data[:, 1], expected_y)


@pytest.mark.parametrize("method", ["linear", "cubic", "nearest"])
def test_resample_bivariate(bivariate_data, method):
    result = resample(bivariate_data, method)
    assert np.allclose(result.grid_shape, (3, 3))
    assert result.n_samples == 9


def test_resample_invalid_dimension():
    sample = Sample(np.zeros((10, 4)), None)
    with pytest.raises(AssertionError):
        resample(sample)
