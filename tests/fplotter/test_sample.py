import pytest
import numpy as np

from fplotter import Sample, Sample2d, Sample3d


@pytest.fixture
def sample_data():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 20)
    z = np.linspace(0, 1, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    W = np.sin(X) * Y + np.cos(Y) * Z + np.tan(Z) * X
    data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel(), W.ravel()))
    return data, (len(x), len(y), len(z))


@pytest.fixture
def sample_object(sample_data):
    data, grid_shape = sample_data
    return Sample(data, grid_shape)


class TestSample:
    def test_init(self, sample_object, sample_data):
        data, grid_shape = sample_data
        assert sample_object.n_samples == data.shape[0]
        assert sample_object.dim == data.shape[1]
        assert np.array_equal(sample_object.grid_shape, grid_shape)
        assert id(sample_object.data) == id(data)

    def test_set_mask(self, sample_object):
        data = sample_object.data.copy()
        mask = np.array([True, False] * (sample_object.n_samples // 2))
        sample_object.set_mask(mask)
        assert np.isnan(sample_object.data[mask, -1]).all()
        assert not np.isnan(sample_object.data[~mask, -1]).any()
        assert np.array_equal(sample_object.data[:, :-1], data[:, :-1])

    def test_reshape_as_grid(self, sample_object):
        grids = sample_object.reshape_as_grid()
        assert isinstance(grids, tuple)
        assert len(grids) == sample_object.dim
        for i, grid in enumerate(grids):
            assert grid.shape == sample_object.grid_shape
            assert np.array_equal(grid.flatten(), sample_object.data[:, i])

    def test_reshape_as_grid_no_shape_raises_error(self, sample_data):
        data, _ = sample_data
        sample_no_grid = Sample(data, None)
        with pytest.raises(AssertionError):
            sample_no_grid.reshape_as_grid()


class TestSample2d:
    def test_init(self):
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        sample = Sample2d(x, y, len(x))
        assert sample.dim == 2
        assert sample.n_samples == 10
        assert np.array_equal(sample.data, np.column_stack((x, y)))
        assert np.array_equal(sample.grid_shape, np.atleast_1d(len(x)))

    def test_reshape_as_grid(self):
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        sample = Sample2d(x, y, 10)
        gx, gy = sample.reshape_as_grid()
        assert np.array_equal(gx, x)
        assert np.array_equal(gy, y)


class TestSample3d:
    def test_init(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * Y + np.cos(Y) * X
        grid_shape = (len(x), len(y))
        sample = Sample3d(X.ravel(), Y.ravel(), Z.ravel(), grid_shape)
        assert sample.dim == 3
        assert sample.n_samples == len(x) * len(y)
        assert np.array_equal(
            sample.data, np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        )
        assert np.array_equal(sample.grid_shape, grid_shape)
