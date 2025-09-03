from __future__ import annotations

from typing import Final
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.tri as tri

FLT_EPS: Final[float] = np.finfo(float).eps


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if not np.isfinite(norm):
        return np.repeat(np.nan, len(v))
    else:
        return v / max(norm, FLT_EPS)


@dataclass
class BaseGeometry(metaclass=ABCMeta):
    point_idx: list[int]
    samples: SampleND
    badness: np.ndarray
    neighbors: list[BaseGeometry]
    normal: np.ndarray = None

    @abstractmethod
    def compute_normal(self) -> None:
        pass

    def midpoints(self) -> np.ndarray:
        # row pairwise midpoint -- magical!
        vertices = self.vertices()[:, :-1]
        i, j = np.triu_indices(len(vertices), k=1)
        return (vertices[i] + vertices[j]) / 2

    def vertices(self) -> np.ndarray:
        return self.samples.data[self.point_idx, :]

    def neighbor_idx(self, geometry: BaseGeometry) -> int | None:
        for i, neighbor in enumerate(self.neighbors):
            if neighbor == geometry:
                return i
        return None

    def __eq__(self, other: BaseGeometry) -> bool:
        if not isinstance(other, BaseGeometry):
            return False
        return (self.point_idx == other.point_idx
                and self.samples == other.samples)

    def max_badness(self) -> float:
        if all(np.isnan(badness) for badness in self.badness):
            return np.nan
        return np.nanmax(self.badness)

    def __repr__(self) -> str:
        with np.printoptions(formatter={'float': '{:.2f}'.format}):
            lines = [
                f'<{self.__class__.__name__} at {hex(id(self))}>',
                f'  @ vertices   : {str(self.vertices()).replace('\n', '')}',
                f'  @ badness    : {self.badness}',
                f'  @ max_badness: {self.max_badness():.1f}',
            ]
        return '\n'.join(lines)

    __str__ = __repr__


class Interval(BaseGeometry):
    def __init__(self, point_idx: list[int], samples: Sample2D):
        super().__init__(point_idx, samples, np.repeat(np.nan, 2), [None] * 2)

    def compute_normal(self) -> None:
        x0, x1 = self.vertices()
        n = np.array([x1[1] - x0[1], x0[0] - x1[0]])
        n = normalize_vector(n)
        self.normal = n if n[-1] > 0 else -n


class Triangle(BaseGeometry):
    def __init__(self, point_idx: list[int], samples: Sample3D):
        super().__init__(point_idx, samples, np.repeat(np.nan, 3), [None] * 3)

    def compute_normal(self) -> None:
        x0, x1, x2 = self.vertices()
        n = np.cross(x1 - x0, x2 - x0)
        n = normalize_vector(n)
        self.normal = n if n[-1] > 0 else -n


class SampleND:
    def __init__(self, data: np.ndarray, auto_geometry: bool = False,
                 auto_normal: bool = False):
        self.data = data
        self.size: int = data.shape[0]
        self.dim: int = data.shape[1]
        self.mask: np.ndarray = np.repeat(False, self.size)  # False -> Plotted

        self.geometry: list[BaseGeometry] | None = None
        self._auto_geometry = auto_geometry
        self._auto_normal = auto_normal
        if auto_geometry:
            self.build_geometry()

        # self.clip_contour: np.ndarray | None = None

    # def get_plot_points(self) -> tuple:
    #     data = self.data.copy()
    #     data[self.mask, -1] = np.nan
    #     return tuple(data.T)

    # def get_clip_contour(self) -> tuple:
    #     return tuple(self.clip_contour.T)

    # def set_clip_contour(self, contour: np.ndarray) -> None:
    #     self.update(contour)
    #     self.clip_contour = contour

    def set_auto_flags(self, auto_geometry: bool, auto_normal: bool) -> None:
        if self._auto_geometry:
            if not self._auto_normal and self.geometry is not None:
                for geometry in self.geometry:
                    geometry.compute_normal()
                self._auto_geometry = auto_geometry
                self._auto_normal = auto_normal
        else:
            self._auto_normal = auto_normal
            self.build_geometry()
            self._auto_geometry = auto_geometry

    def build_geometry(self) -> None:
        pass

    def add_data(self, data: np.ndarray) -> None:
        if len(data) == 0:
            return
        self.data = np.vstack((self.data, data))
        self.mask = np.concatenate((self.mask, np.repeat(False, data.shape[0])))
        if self._auto_geometry:
            self.build_geometry()

    def __repr__(self) -> str:
        lines = [
            f'<{self.__class__.__name__} at {hex(id(self))}>',
            f'  @ dim     : {self.data.shape[1]}',
            f'  @ points  : {self.data.shape[0]}'
        ]
        if self.geometry is None or len(self.geometry) == 0:
            lines.append(f'  @ geometry: None')
        else:
            lines.append(f'  @ geometry: {self.geometry[0].__class__.__name__}')
        return '\n'.join(lines)

    __str__ = __repr__


class Sample2D(SampleND):
    def __init__(self, data: np.ndarray, auto_geometry: bool = False,
                 auto_normal: bool = False):
        super().__init__(data, auto_geometry, auto_normal)

    def build_geometry(self) -> None:
        idx = np.argsort(self.data[:, 0])
        self.data = self.data[idx]
        self.mask = self.mask[idx]

        intervals = []
        for i in range(self.data.shape[0] - 1):
            interval = Interval([i, i + 1], self)
            if i > 0:
                interval.neighbors[0] = intervals[-1]
                intervals[-1].neighbors[1] = interval
            if self._auto_normal:
                interval.compute_normal()
            intervals.append(interval)
        self.geometry = intervals


class Sample3D(SampleND):
    def __init__(self, data: np.ndarray, auto_geometry: bool = False,
                 auto_normal: bool = False):
        super().__init__(data, auto_geometry, auto_normal)
        self.mtri: tri.Triangulation | None = None

    def build_geometry(self) -> None:
        mtri = tri.Triangulation(self.data[:, 0], self.data[:, 1])
        triangles = []
        for mtri_triangle in mtri.triangles:
            triangle = Triangle(mtri_triangle.tolist(), self)
            if self._auto_normal:
                triangle.compute_normal()
            triangles.append(triangle)

        for i, mpl_neighbor in enumerate(mtri.neighbors):
            triangle = triangles[i]
            for j, idx in enumerate(mpl_neighbor):
                if idx == -1:
                    continue
                triangle.neighbors[j] = triangles[idx]

        self.geometry = triangles
        self.mtri = mtri
