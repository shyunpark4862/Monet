import pytest
import numpy as np

from fplotter import Sample, Sample2d, Sample3d, sample_uniform


def univariate_func(x):
    return np.sin(x)


def bivariate_func(x, y):
    return np.sin(x) * y + np.cos(y) * x


def trivariate_func(x, y, z):
    return np.sin(x) * y + np.cos(y) * z + np.tan(z) * x


@pytest.fixture
def sample_data():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 20)
    z = np.linspace(0, 1, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    W = trivariate_func(X, Y, Z)
    data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel(), W.ravel()))
    return data, np.array([len(x), len(y), len(z)])


class TestSample:
    def test_init(self, sample_data):
        data, grid_shape = sample_data
        sample = Sample(data, grid_shape)
        assert sample.n_samples == data.shape[0]
        assert sample.dim == data.shape[1]
        assert np.array_equal(sample.grid_shape, grid_shape)
        assert id(sample.data) == id(data)

    def test_set_mask(self, sample_data):
        data, grid_shape = sample_data
        sample = Sample(data.copy(), grid_shape)
        mask = np.array([True, False] * (sample.n_samples // 2))
        sample.set_mask(mask)
        assert np.isnan(sample.data[mask, -1]).all()
        assert not np.isnan(sample.data[~mask, -1]).any()
        assert np.array_equal(sample.data[:, :-1], data[:, :-1])

    def test_reshape_as_grid(self, sample_data):
        data, grid_shape = sample_data
        sample = Sample(data, grid_shape)
        grids = sample.reshape_as_grid()
        assert isinstance(grids, tuple)
        assert len(grids) == sample.dim
        for i, grid in enumerate(grids):
            assert np.array_equal(grid.shape, sample.grid_shape)
            assert np.array_equal(grid.flatten(), sample.data[:, i])

    def test_reshape_as_grid_no_shape_raises_error(self, sample_data):
        data, _ = sample_data
        sample_no_grid = Sample(data, None)
        with pytest.raises(AssertionError):
            sample_no_grid.reshape_as_grid()


class TestSample2d:
    def test_init(self):
        x = np.linspace(0, 1, 10)
        y = univariate_func(x)
        sample = Sample2d(x, y, len(x))
        assert sample.dim == 2
        assert sample.n_samples == 10
        assert np.array_equal(sample.data, np.column_stack((x, y)))
        assert np.array_equal(sample.grid_shape, np.atleast_1d(len(x)))

    def test_reshape_as_grid(self):
        x = np.linspace(0, 1, 10)
        y = univariate_func(x)
        sample = Sample2d(x, y, 10)
        gx, gy = sample.reshape_as_grid()
        assert np.array_equal(gx, x)
        assert np.array_equal(gy, y)


class TestSample3d:
    def test_init(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y)
        Z = bivariate_func(X, Y)
        grid_shape = (len(x), len(y))
        sample = Sample3d(X.ravel(), Y.ravel(), Z.ravel(), grid_shape)
        assert sample.dim == 3
        assert sample.n_samples == len(x) * len(y)
        assert np.array_equal(
            sample.data, np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        )
        assert np.array_equal(sample.grid_shape, grid_shape)


class TestSampleUniform:
    def test_univariate_sampling(self):
        n_samples = 11
        xbound = (0, 10)
        sample = sample_uniform(univariate_func, n_samples, xbound)
        assert isinstance(sample, Sample2d)
        assert sample.n_samples == n_samples
        assert sample.dim == 2

        x = np.linspace(xbound[0], xbound[1], n_samples)
        y = univariate_func(x)
        assert np.array_equal(sample.data[:, 0], x)
        assert np.array_equal(sample.data[:, 1], y)
        assert np.array_equal(sample.grid_shape, np.atleast_1d(n_samples))

    def test_bivariate_sampling(self):
        n_samples = (5, 6)
        xbound, ybound = (0, 4), (0, 5)
        sample = sample_uniform(bivariate_func, n_samples, xbound, ybound)
        assert isinstance(sample, Sample3d)
        assert sample.n_samples == n_samples[0] * n_samples[1]
        assert sample.dim == 3

        x = np.linspace(xbound[0], xbound[1], n_samples[0])
        y = np.linspace(ybound[0], ybound[1], n_samples[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = bivariate_func(X, Y)
        data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        assert np.array_equal(sample.data, data)
        assert np.array_equal(sample.grid_shape, np.array(n_samples))
