from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.spatial import Delaunay

from sample import Sample, Sample2d, Sample3d

FLT_EPS: float = np.finfo(float).eps


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if not np.isfinite(norm):
        return np.repeat(np.nan, len(v))
    else:
        return v / max(norm, FLT_EPS)


@dataclass
class Geometry(ABC):
    points: tuple[int, ...]
    data: np.ndarray
    badness: list[float]
    neighbors: list[Geometry]
    normal: np.ndarray = None

    @abstractmethod
    def compute_normal(self) -> None:
        pass

    @abstractmethod
    def midpoints(self) -> np.ndarray:
        pass

    def vertices(self) -> np.ndarray:
        return self.data[self.points, :]

    def neighbor_idx(
            self,
            geometry: Geometry
    ) -> int | None:
        for i, neighbor in enumerate(self.neighbors):
            if neighbor == geometry:
                return i
        return None

    def max_badness(self) -> float:
        if np.isnan(self.badness).all():
            return np.nan
        return np.nanmax(self.badness)


class Interval(Geometry):
    def __init__(
            self,
            points: list[int],
            data: np.ndarray
    ):
        super().__init__(points, data, np.repeat(np.nan, 3), [None] * 2)

    def midpoints(self) -> float:
        x0, x1 = self.vertices()[:, 0]
        return (x0 + x1) / 2

    def compute_normal(self) -> None:
        x0, x1 = self.vertices()
        n = np.array([x1[1] - x0[1], x0[0] - x1[0]])
        n = normalize_vector(n)
        self.normal = n if n[-1] > 0 else -n


class Triangle(Geometry):
    def __init__(
            self,
            points: list[int],
            data: np.ndarray
    ):
        super().__init__(points, data, np.repeat(np.nan, 4), [None] * 3)

    def midpoints(self) -> np.ndarray:
        (x0, y0), (x1, y1), (x2, y2) = self.vertices()[:, :-1]
        xmid0, xmid1, xmid2 = (x0 + x1) / 2, (x1 + x2) / 2, (x2 + x0) / 2
        ymid0, ymid1, ymid2 = (y0 + y1) / 2, (y1 + y2) / 2, (y2 + y0) / 2
        return np.array(((xmid0, ymid0), (xmid1, ymid1), (xmid2, ymid2)))

    def compute_normal(self) -> None:
        x0, x1, x2 = self.vertices()
        n = np.cross(x1 - x0, x2 - x0)
        n = normalize_vector(n)
        self.normal = n if n[-1] > 0 else -n


class Refiner(ABC):
    def __init__(
            self,
            func: Callable[[np.ndarray, ...], np.ndarray],
            sample: Sample,
            contour_levels: np.ndarray,
            contour_only: bool,
            threshold: float,
            n_iters: int
    ):
        self.func = func
        self.data: np.ndarray = sample.data.copy()
        self.contour_levels = contour_levels
        self.contour_only = contour_only
        self.threshold = threshold
        self.n_iters = n_iters
        self.geometry: list[Interval] | None = None
        self._early_exit: bool = False
        self._n_new_data: int = 0

    def run(self) -> Sample:
        self._early_exit = False
        for _ in range(self.n_iters):
            if self._early_exit:
                break
            self._build_geometry()
            self._compute_badness()
            self._refine()

        return Sample(self.data, None)

    @abstractmethod
    def _build_geometry(self) -> None:
        pass

    def _compute_badness(self) -> None:
        for geometry in self.geometry:
            if not self.contour_only:
                for i, neighbor in enumerate(geometry.neighbors):
                    if not np.isnan(geometry.badness[i]) or neighbor is None:
                        continue
                    curvature = self._compute_curvature(geometry, neighbor)
                    j = neighbor.neighbor_idx(geometry)
                    geometry.badness[i] = neighbor.badness[j] = curvature

            values = geometry.vertices()[:, -1]
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            if ((self.contour_levels > (vmin - FLT_EPS))
                & (self.contour_levels < (vmax + FLT_EPS))).any():
                geometry.badness[-1] = np.inf
            else:
                geometry.badness[-1] = -np.inf

    @staticmethod
    def _compute_curvature(
            geometry: Geometry,
            neighbor: Geometry
    ) -> float:
        v, w = geometry.normal, neighbor.normal
        angle = np.arccos(min(np.dot(v, w), 1))
        return angle if not np.isnan(angle) else -np.inf

    def _refine(self) -> None:
        coords = []
        for geometry in self.geometry:
            if geometry.max_badness() > self.threshold:
                coords.append(geometry.midpoints())
        if not coords:
            self._early_exit = True
            return
        coords = np.vstack(coords)
        values = np.atleast_2d(self.func(*coords.T)).T
        data = np.hstack((coords, values))
        self.data = np.vstack((self.data, data))
        self._n_new_data = data.shape[0]


class UnivariateRefiner(Refiner):
    def __init__(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            sample: Sample2d,
            contour_levels: np.ndarray,
            contour_only: bool,
            threshold: float,
            n_iters: int
    ):
        super().__init__(
            func, sample, contour_levels, contour_only, threshold, n_iters
        )

    def _build_geometry(self) -> None:
        self.data = self.data[np.argsort(self.data[:, 0])]
        intervals = []
        for i in range(self.data.shape[0] - 1):
            interval = Interval([i, i + 1], self.data)
            if i > 0:
                interval.neighbors[0] = intervals[-1]
                intervals[-1].neighbors[1] = interval
            if not self.contour_only:
                interval.compute_normal()
            intervals.append(interval)
        self.geometry = intervals


class BivariateRefiner(Refiner):
    def __init__(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            sample: Sample3d,
            contour_levels: np.ndarray,
            contour_only: bool,
            threshold: float,
            n_iters: int
    ):
        super().__init__(
            func, sample, contour_levels, contour_only, threshold, n_iters
        )
        self._delaunay: Delaunay | None = None

    def _build_geometry(self) -> None:
        if self._delaunay is None:
            self._delaunay = Delaunay(self.data[:, :-1], incremental=True)
        else:
            self._delaunay.add_points(self.data[-self._n_new_data:, :-1])
        triangles = []
        for point_idx in self._delaunay.simplices:
            triangle = Triangle(point_idx.tolist(), self.data)
            if not self.contour_only:
                triangle.compute_normal()
            triangles.append(triangle)
        for i, neighbors in enumerate(self._delaunay.neighbors):
            triangle = triangles[i]
            for j, idx in enumerate(neighbors):
                if idx == -1:
                    continue
                triangle.neighbors[j] = triangles[idx]
        self.geometry = triangles


def refine_univariate(
        func: Callable[[np.ndarray], np.ndarray],
        sample: Sample2d,
        contour_levels: np.ndarray,
        contour_only: bool,
        threshold: float,
        n_iters: int
) -> Sample2d:
    return UnivariateRefiner(
        func, sample, contour_levels, contour_only, threshold, n_iters
    ).run()


def refine_bivariate(
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        sample: Sample3d,
        contour_levels: np.ndarray,
        contour_only: bool,
        threshold: float,
        n_iters: int
) -> Sample3d:
    return BivariateRefiner(
        func, sample, contour_levels, contour_only, threshold, n_iters
    ).run()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    f = lambda x, y: np.sin(np.sqrt(x ** 2 + y ** 2))
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    sample = Sample3d(X.ravel(), Y.ravel(), Z.ravel(), (10, 10))
    sample = refine_bivariate( f, sample, [0.5, 0.75, -0.75], False, 0.1745, 5)
    # sample = resample_bivariate(sample, "linear")
    plt.contourf(X, Y, Z)
    plt.scatter(*sample.data[:, :-1].T, alpha=0.3, c='grey', s=3)
    plt.show()
