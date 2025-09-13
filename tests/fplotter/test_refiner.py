import numpy as np
from dataclasses import dataclass

from fplotter.sampler import Sample2d, Sample3d
from fplotter.refiner import FLT_EPS, _normalize, _Interval, _Triangle, \
    _build_triangles, _build_intervals, _compute_badness, _refine_mesh, \
    refine, _compute_total_length, _compute_total_area
from test_functions import univariate_func, bivariate_func


class TestHelperFunctions:
    def test_normalize_standard_vector(self):
        v = np.array([3.0, 4.0])
        expected = np.array([0.6, 0.8])
        assert np.allclose(_normalize(v), expected)

    def test_normalize_zero_vector(self):
        v = np.array([0.0, 0.0])
        expected = np.array([0.0, 0.0])
        assert np.allclose(_normalize(v), expected)

    def test_normalize_non_finite_vector(self):
        v = np.array([np.inf, np.nan])
        result = _normalize(v)
        assert np.isnan(result).all()


class TestMeshElements:
    def setup_method(self):
        self.data_2d = np.array([[0., 0.], [1., 1.], [2., 0.]])
        self.data_3d = np.array([[0., 0., 0.], [1., 0., 1.], [0., 1., 1.]])
        self.interval = _Interval((0, 1), self.data_2d)
        self.triangle = _Triangle((0, 1, 2), self.data_3d)

    def test_interval_initialization(self):
        assert self.interval.points == (0, 1)
        assert np.isnan(self.interval.badness).all()
        assert self.interval.neighbors == [None, None]

    def test_interval_vertices(self):
        expected = np.array([[0., 0.], [1., 1.]])
        assert np.allclose(self.interval.vertices(), expected)

    def test_interval_midpoint(self):
        assert self.interval.midpoints() == 0.5

    def test_interval_compute_normal(self):
        self.interval.compute_normal()
        expected = np.array([-1.0, 1.0]) / np.sqrt(2)
        assert np.allclose(self.interval.normal, expected)

    def test_triangle_vertices(self):
        assert np.allclose(self.triangle.vertices(), self.data_3d)

    def test_triangle_midpoints(self):
        expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        assert np.allclose(self.triangle.midpoints(), expected)

    def test_triangle_compute_normal(self):
        self.triangle.compute_normal()
        expected = np.array([-1., -1., 1.]) / np.sqrt(3)
        assert np.allclose(self.triangle.normal, expected)

    def test_max_badness(self):
        self.interval.badness = [0.1, np.nan, 0.5]
        assert self.interval.max_badness() == 0.5
        self.interval.badness = [np.nan, np.nan]
        assert np.isnan(self.interval.max_badness())

    def test_interval_compute_area(self):
        assert np.allclose(self.interval.area, 1.0)

    def test_triangle_compute_area(self):
        assert np.allclose(self.triangle.area, 0.5)


class TestAreaHelpers:
    def test_compute_total_length(self):
        x = np.linspace(0, 10, 11)
        y = univariate_func(x)
        data = np.column_stack((x, y))
        assert np.allclose(_compute_total_length(data), 10)

    def test_compute_total_area(self):
        x = np.linspace(0, 10, 11)
        y = np.linspace(-5, 0, 5)
        X, Y = np.meshgrid(x, y)
        Z = bivariate_func(X, Y)
        data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        assert np.allclose(_compute_total_area(data), 50)


class TestMeshBuilding:
    def test_build_intervals(self):
        data = np.array([[0., 0.], [1., 1.], [2., 0.]])
        intervals = _build_intervals(data, True)
        assert len(intervals) == 2
        assert intervals[0].neighbors[1] == intervals[1]
        assert intervals[1].neighbors[0] == intervals[0]
        assert intervals[0].normal is not None

    def test_build_triangles(self):
        data = np.array(
            [[0., 0., 0.], [1., 0., 1.], [0., 1., 1.], [1.1, 1.0, 2.]]
        )
        triangles, _ = _build_triangles(data, True, None, 0)
        assert len(triangles) == 2
        assert triangles[1] in triangles[0].neighbors
        assert triangles[0].normal is not None

    def test_build_triangles_incremental(self):
        data = np.array(
            [[0., 0., 0.], [1., 0., 1.], [0., 1., 1.], [1.1, 1.0, 2.]]
        )
        triangles, delaunay = _build_triangles(
            data, True, None, 0
        )
        assert len(triangles) == 2

        new_data = np.array([[0.5, 0.5, 0.5]])
        data = np.vstack((data, new_data))
        triangles_updated, _ = _build_triangles(
            data, True, delaunay, 1
        )
        assert len(triangles_updated) == 4


class TestBadnessCalculation:
    def test_compute_badness_contour_intersection(self):
        data = np.array([[0., 0.], [1., 2.]])
        interval = _Interval((0, 1), data)
        contour_levels = np.array([1.0])
        _compute_badness([interval], contour_levels, True)
        assert interval.badness[-1] == np.inf

    def test_compute_badness_no_contour_intersection(self):
        data = np.array([[0., 0.], [1., 2.]])
        interval = _Interval((0, 1), data)
        contour_levels = np.array([3.0])
        _compute_badness([interval], contour_levels, True)
        assert interval.badness[-1] == -np.inf

    def test_compute_badness_with_curvature(self):
        data = np.array([[0., 0.], [1., 0.], [2., 1.]])
        intervals = _build_intervals(data, True)
        _compute_badness(intervals, np.array([]), False)
        # n0 = [0, 1], n1 = [-1/sqrt(2), 1/sqrt(2)]
        # dot(n0, n1) = 1/sqrt(2)
        # arccos(1/sqrt(2)) = pi/4
        expected_angle = np.pi / 4
        assert np.isclose(intervals[0].badness[1], expected_angle)
        assert intervals[0].badness[1] == intervals[1].badness[0]


class TestRefineMesh:
    def test_refine_mesh_adds_points(self):
        data = np.array([[0., 0.], [2., 4.]])
        interval = _Interval((0, 1), data)
        interval.badness[-1] = np.inf
        refined_data, n_new = _refine_mesh(
            univariate_func, data, [interval], 0.1, 2.0, 1e-6
        )
        expected_new_point = np.array([1.0, univariate_func(1.0)])
        assert n_new == 1
        assert refined_data.shape[0] == 3
        assert np.allclose(refined_data[-1], expected_new_point)

    def test_refine_mesh_early_exit(self):
        data = np.array([[0., 0.], [2., 4.]])
        interval = _Interval((0, 1), data)
        interval.badness = [0.01, 0.01, -np.inf]
        refined_data, n_new = _refine_mesh(
            univariate_func, data, [interval], 0.1, 2.0, 1e-6
        )
        assert n_new == 0
        assert np.allclose(refined_data, data)

    def test_refine_mesh_skips_small_area(self):
        data = np.array([[0., 0.], [0.001, 0.001 ** 2]])
        interval = _Interval((0, 1), data)
        interval.badness[-1] = np.inf
        refined_data, n_new = _refine_mesh(
            univariate_func, data, [interval], 0.1, 100, 1e-4
        )
        assert n_new == 0
        assert len(refined_data) == 2


class TestRefineIntegration:
    def test_refine_univariate_by_curvature(self):
        x = np.linspace(0, np.pi, 5)
        sample = Sample2d(x, univariate_func(x), len(x))
        result = refine(
            univariate_func, sample, np.array([]), False, 0.5, 1
        )
        assert len(result.data) > sample.n_samples
        assert result.grid_shape is None

    def test_refine_bivariate_by_contour(self):
        x = np.linspace(0, np.pi, 5)
        y = np.linspace(0, np.pi, 10)
        X, Y = np.meshgrid(x, y)
        Z = bivariate_func(X, Y)
        sample = Sample3d(X.ravel(), Y.ravel(), Z.ravel(), (len(x), len(y)))
        result = refine(
            bivariate_func, sample, np.array([0.5]), True, 0.1, 2
        )
        assert len(result.data) > sample.n_samples
