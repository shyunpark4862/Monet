import pytest
import numpy as np
import numpy.ma as ma

import fplotter.sampler as sampler
from test_functions import univariate_func, bivariate_func, trivariate_func


@pytest.fixture
def sample_data():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 20)
    z = np.linspace(0, 1, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    W = trivariate_func(X, Y, Z)
    data = ma.column_stack((X.ravel(), Y.ravel(), Z.ravel(), W.ravel()))
    return data, np.array([len(x), len(y), len(z)])


class TestSample:
    def test_init(self, sample_data):
        data, grid_shape = sample_data
        result = sampler.Sample(data, grid_shape)
        assert result.n_samples == data.shape[0]
        assert result.dim == data.shape[1]
        assert np.allclose(result.grid_shape, grid_shape)
        assert id(result.data) == id(data)

    def test_set_mask(self, sample_data):
        data, grid_shape = sample_data
        result = sampler.Sample(data, grid_shape)
        mask = np.repeat(False, result.n_samples)
        mask[0] = True
        result.set_mask(mask)

        assert not (result.data.mask[:,:-1].any())
        assert result.data.mask[0, -1]
        assert not (result.data.mask[1:,-1].any())

    def test_reshape_as_grid(self, sample_data):
        data, grid_shape = sample_data
        result = sampler.Sample(data, grid_shape)
        result_grids = result.reshape_as_grid(False)
        assert isinstance(result_grids, tuple)
        assert len(result_grids) == result.dim
        for i, grid in enumerate(result_grids):
            assert np.allclose(grid.shape, result.grid_shape)
            assert np.allclose(grid.flatten(), result.data[:, i])

    def test_reshape_as_grid_mask_applied(self, sample_data):
        data, grid_shape = sample_data
        data.mask[:10, -1] = True
        result = sampler.Sample(data, grid_shape)
        result_grids = result.reshape_as_grid(True)
        assert isinstance(result_grids, tuple)
        for i, grid in enumerate(result_grids):
            if i == data.shape[1] - 1:
                assert np.isnan(grid.ravel()[:10]).all()
            else:
                assert np.isfinite(grid).all()

    def test_reshape_as_grid_no_shape_raises_error(self, sample_data):
        data, _ = sample_data
        result = sampler.Sample(data, None)
        with pytest.raises(AssertionError):
            result.reshape_as_grid(False)


class TestSample2d:
    def test_init(self):
        x = np.linspace(0, 1, 10)
        y = univariate_func(x)
        result = sampler.Sample2d(x, y, len(x))
        assert result.dim == 2
        assert result.n_samples == 10
        assert np.allclose(result.data, np.column_stack((x, y)))
        assert np.allclose(result.grid_shape, np.atleast_1d(len(x)))

    def test_reshape_as_grid(self):
        x = np.linspace(0, 1, 10)
        y = univariate_func(x)
        result = sampler.Sample2d(x, y, 10)
        result_grid = result.reshape_as_grid(True)
        assert np.allclose(result_grid[0], x)
        assert np.allclose(result_grid[1], y)


class TestSample3d:
    def test_init(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y)
        Z = bivariate_func(X, Y)
        grid_shape = (len(x), len(y))
        result = sampler.Sample3d(X.ravel(), Y.ravel(), Z.ravel(), grid_shape)
        assert result.dim == 3
        assert result.n_samples == len(x) * len(y)
        assert np.allclose(
            result.data, np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        )
        assert np.allclose(result.grid_shape, grid_shape)


class TestSampleUniform:
    def test_univariate_sampling(self):
        n_samples = 11
        xbound = (0, 10)
        result = sampler.sample_uniform(univariate_func, n_samples, xbound)
        assert isinstance(result, sampler.Sample2d)
        assert result.n_samples == n_samples
        assert result.dim == 2

        expected_x = np.linspace(xbound[0], xbound[1], n_samples)
        expected_y = univariate_func(expected_x)
        assert np.allclose(result.data[:, 0], expected_x)
        assert np.allclose(result.data[:, 1], expected_y)
        assert np.allclose(result.grid_shape, np.atleast_1d(n_samples))

    def test_bivariate_sampling(self):
        n_samples = (5, 6)
        xbound, ybound = (0, 4), (0, 5)
        result = sampler.sample_uniform(bivariate_func, n_samples, xbound, ybound)
        assert isinstance(result, sampler.Sample3d)
        assert result.n_samples == n_samples[0] * n_samples[1]
        assert result.dim == 3

        x = np.linspace(*xbound, n_samples[0])
        y = np.linspace(*ybound, n_samples[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = bivariate_func(X, Y)
        expected_data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        assert np.allclose(result.data, expected_data)
        assert np.allclose(result.grid_shape, np.array(n_samples))
