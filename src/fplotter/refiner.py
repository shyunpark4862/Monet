"""
The MIT License

Copyright (c) 2025 SangHyun Park

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final

import numpy as np
from scipy.spatial import Delaunay

from .sampler import Sample2d, Sample3d

FLT_EPS: Final[float] = np.finfo(float).eps  # Floating point epsilon

""" CLASSES FOR MESH """


class _MeshElement(ABC):
    """
    An abstract base class for a geometric element in a mesh.

    This class serves as a container for the properties of a mesh element,
    such as its vertices, neighbors, normal vector, and a "badness" metric
    used for refinement.

    Parameters
    ----------
    points : tuple[int, ...]
        Indices of the vertices that form this mesh element, referencing the
        main data array.
    data : np.ndarray
        A reference to the full dataset array containing all sample points.
    badness : list[float]
        A list storing the badness metric.
    neighbors : list[_MeshElement]
        A list of neighboring mesh elements.

    Attributes
    ----------
    normal : np.ndarray or None
        The normal vector of the element.

    area : float or None
        The area (2D) or length (1D) of the element.
    """

    def __init__(
            self,
            points: tuple[int, ...],
            data: np.ndarray,
            badness: list[float],
            neighbors: list[_MeshElement]
    ):

        self.points = points
        self.data = data
        self.badness = badness
        self.neighbors = neighbors
        self.normal: np.ndarray | None = None
        self.area: float | None = None

        self.compute_area()

    @abstractmethod
    def compute_normal(self) -> None:
        pass

    @abstractmethod
    def compute_area(self) -> float:
        pass

    @abstractmethod
    def midpoints(self) -> np.ndarray:
        pass

    def vertices(self) -> np.ndarray:
        """
        Returns the vertices of a mesh element as an array.

        Returns
        -------
        numpy.ndarray
            An array containing the vertices of the shape.
        """
        return self.data[self.points, :]

    def neighbor_idx(
            self,
            mesh: _MeshElement
    ) -> int | None:
        """
        Finds the index of a given neighbor in the neighbors list.

        Parameters
        ----------
        mesh : _MeshElement
            The neighboring mesh element to find.

        Returns
        -------
        int or None
            The index of the neighbor, or None if it is not found.
        """
        for i, neighbor in enumerate(self.neighbors):
            if neighbor == mesh:
                return i
        return None

    def max_badness(self) -> float:
        """
        Calculates the maximum badness value for this element.

        Returns
        -------
        float
            The maximum badness value, ignoring NaNs. Returns NaN if all
            badness values are NaN.
        """
        if np.isnan(self.badness).all():
            return np.nan
        return np.nanmax(self.badness)


class _Interval(_MeshElement):
    """
    Represents a 1D interval (a line segment) in the mesh.
    """

    def __init__(
            self,
            points: tuple[int, int],
            data: np.ndarray
    ):
        super().__init__(points, data, np.repeat(np.nan, 3), [None] * 2)

    def compute_normal(self) -> None:
        """
        Computes and assigns the normal vector for a 2D interval (line segment).

        For a line segment from point ``x0 = (x0, y0)`` to ``x1 = (x1, y1)``, 
        the normal vector ``n`` is computed as ``n = (y1 - y0, -(x1 - x0))``.
        The resulting normal vector is then normalized to unit length and its 
        direction is adjusted to ensure the y-component is positive.
        """
        x0, x1 = self.vertices()
        n = np.array([x1[1] - x0[1], x0[0] - x1[0]])
        n = _normalize(n)
        self.normal = n if n[-1] > 0 else -n

    def compute_area(self) -> float:
        """
        Computes the area (length) of an interval.
        """
        x0, x1 = self.vertices()[:, 0]
        self.area = x1 - x0

    def midpoints(self) -> float:
        """
        Computes the midpoint of the interval.

        Returns
        -------
        float
            Midpoint of the interval.
        """
        x0, x1 = self.vertices()[:, 0]
        return (x0 + x1) / 2


class _Triangle(_MeshElement):
    """
    Represents a 2D triangle in the mesh.
    """

    def __init__(
            self,
            points: tuple[int, int, int],
            data: np.ndarray
    ):
        super().__init__(points, data, np.repeat(np.nan, 4), [None] * 3)

    def compute_normal(self) -> None:
        """
        Computes and assigns the normal vector for a 3D triangle.

        This method calculates the normal vector of a triangle defined by its 
        vertices. The normal is derived using the cross product of two edges of 
        the triangle. The normal vector ``n`` is calculated using the formula 
        ``n = (x2 - x0) × (x2 - x0)`` where ``x0``, ``x1``, ``x2`` are the 
        vertices of the triangle. The resulting normal vector is then normalized 
        to unit length and its direction is adjusted to ensure the z-component 
        is positive.
        """
        x0, x1, x2 = self.vertices()
        n = np.cross(x1 - x0, x2 - x0)
        n = _normalize(n)
        self.normal = n if n[-1] > 0 else -n

    def compute_area(self) -> float:
        """
        Computes and assigns the area of a triangle.

        This method computes the area of a triangle using Shoelace formula. For
        more details, see:

        - Shoelace formula : https://en.wikipedia.org/wiki/Shoelace_formula.
        """
        (x0, y0), (x1, y1), (x2, y2) = self.vertices()[:, :-1]
        self.area = abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) / 2

    def midpoints(self) -> np.ndarray:
        """
        Computes midpoints of the edges of a triangle.

        Returns
        -------
        np.ndarray of shape (3, 2)
            A np.ndarray containing the midpoints of the edges. Each row
            corresponds to a midpoint in the format (x, y).
        """
        (x0, y0), (x1, y1), (x2, y2) = self.vertices()[:, :-1]
        xmid0, xmid1, xmid2 = (x0 + x1) / 2, (x1 + x2) / 2, (x2 + x0) / 2
        ymid0, ymid1, ymid2 = (y0 + y1) / 2, (y1 + y2) / 2, (y2 + y0) / 2
        return np.array(((xmid0, ymid0), (xmid1, ymid1), (xmid2, ymid2)))


def _normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to a unit vector.

    Parameters
    ----------
    v : np.ndarray
        The input vector to normalize.

    Returns
    -------
    np.ndarray
        The normalized unit vector. Returns a vector of NaNs if the norm is
        not finite.
    """
    norm = np.linalg.norm(v)
    if not np.isfinite(norm):
        return np.repeat(np.nan, len(v))
    else:
        return v / max(norm, FLT_EPS)


""" MAIN ROUTINES """


def refine(
        func: Callable[[np.ndarray], np.ndarray] |
              Callable[[np.ndarray, np.ndarray], np.ndarray],
        sample: Sample2d | Sample3d,
        contour_levels: np.ndarray,
        contour_only: bool,
        theta: float,
        n_iters: int
) -> Sample2d | Sample3d:
    """
    Adaptively refines a sample mesh by adding points in areas of interest.

    This function iteratively improves the resolution of a function sample by
    adding new points where the mesh is considered 'bad'. A mesh element is
    marked as bad if it is intersected by a contour level or if the local
    curvature exceeds a specified threshold ``theta``.

    The refinement loop consists of three main phases:
    
    1. Building the mesh structure (triangulation/intervals)
    2. Computing badness metrics for each mesh element
    3. Refining the mesh by adding new points based on badness criteria

    Parameters
    ----------
    func : Callable
        The univariate or bivariate function that is being sampled.
    sample : Sample2d or Sample3d
        An object containing the initial sample points.
    contour_levels : np.ndarray
        Contour levels to refine around.
    contour_only : bool
        If True, refinement is based only on contour line intersections. If
        False, refinement is based on both contours and local curvature.
    theta : float
        The badness threshold in radians. Mesh elements with a maximum badness
        (curvature angle) greater than this value will be refined.
    n_iters : int
        The number of refinement iterations to perform.

    Returns
    -------
    Sample2d or Sample3d
        A new sample object containing both the original and the new refined
        points.
        
    Notes
    -----
    Since the refined sample points are not on a regular rectangular grid, the
    output sample will not have a valid ``grid_shape`` attribute. The sample
    must be resampled onto a regular grid before plotting.

    This function does not modify the input ``sample``, it only returns a new
    sample object.

    The current implementation rebuilds the mesh structure in each refinement 
    iteration, which is inefficient. Performance could be improved by locally 
    updating the mesh structure and caching badness values for unchanged
    regions. This would require implementing local Delaunay triangulation
    updates using a modified Bowyer-Watson algorithm, rather than relying on
    ``scipy.spatial.Delaunay`` which only supports full rebuilds. The adaptive
    library (https://github.com/python-adaptive/adaptive) demonstrates this
    approach.
    """
    data = sample.data
    if sample.dim == 2:
        return _refine_univariate(
            func, data, contour_levels, contour_only, theta, n_iters
        )
    elif sample.dim == 3:
        return _refine_bivariate(
            func, data, contour_levels, contour_only, theta, n_iters
        )
    else:
        assert False, "Invalid sample dimension"


def _refine_univariate(
        func: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        contour_levels: np.ndarray,
        contour_only: bool,
        theta: float,
        n_iters: int
) -> Sample2d:
    """
    Helper function to perform iterative refinement for univariate (2D) data.

    Parameters
    ----------
    func : Callable
        The univariate function being sampled.
    data : np.ndarray
        The array of (x, y) sample points.
    contour_levels : np.ndarray
        Contour levels to refine around.
    contour_only : bool
        Whether to refine based only on contours.
    theta : float
        The badness (curvature) threshold.
    n_iters : int
        The number of refinement iterations.

    Returns
    -------
    Sample2d
        A new sample object with the refined data.
    """
    total_length = _compute_total_length(data)
    early_exit = False
    for _ in range(n_iters):
        if early_exit:
            break
        intervals = _build_intervals(data, not contour_only)
        _compute_badness(intervals, contour_levels, contour_only)
        data, n_new_samples = _refine_mesh(
            func, data, intervals, theta, total_length, 1e-6
        )
        early_exit = n_new_samples == 0
    return Sample2d(*data.T, None)


def _refine_bivariate(
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        data: np.ndarray,
        contour_levels: np.ndarray,
        contour_only: bool,
        theta: float,
        n_iters: int
) -> Sample3d:
    """
    Helper function to perform iterative refinement for bivariate (3D) data.

    Parameters
    ----------
    func : Callable
        The bivariate function being sampled.
    data : np.ndarray
        The array of (x, y, z) sample points.
    contour_levels : np.ndarray
        Contour levels to refine around.
    contour_only : bool
        Whether to refine based only on contours.
    theta : float
        The badness (curvature) threshold.
    n_iters : int
        The number of refinement iterations.

    Returns
    -------
    Sample3d
        A new sample object with the refined data.
    """
    total_area = _compute_total_area(data)
    delaunay = None
    n_new_samples = 0
    early_exit = False
    for _ in range(n_iters):
        if early_exit:
            break
        triangles, delaunay = _build_triangles(
            data, not contour_only, delaunay, n_new_samples
        )
        _compute_badness(triangles, contour_levels, contour_only)
        data, n_new_samples = _refine_mesh(
            func, data, triangles, theta, total_area, 1e-6
        )
        early_exit = n_new_samples == 0
    return Sample3d(*data.T, None)


def _compute_total_length(data: np.ndarray) -> float:
    """
    Computes the total sampling interval length.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, 2)
        A 2D array containing pairs of (x, y) sampled points, where x 
        coordinates are assumed to be sorted in ascending order.

    Returns
    -------
    float
        The total length of the sampling interval.
    """
    return data[-1, 0] - data[0, 0]


def _compute_total_area(data: np.ndarray) -> float:
    """
    Computes the total area of the sampling region.

    This function calculates the total area of the rectangular region where
    the bivariate function is sampled. The area is determined by the bounds
    of the sampling points in both x and y dimensions. The input data is
    expected to contain the boundary points defining the sampling region.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, 3)
        A 2D array containing triples of (x, y, z) sampled points. The x and y
        coordinates are expected to be raveled from a ``np.meshgrid()``.

    Returns
    -------
    float
        The total area of the sampling region.
    """
    return (data[-1, 0] - data[0, 0]) * (data[-1, 1] - data[0, 1])


def _build_intervals(
        data: np.ndarray,
        compute_normal: bool
) -> list[_Interval]:
    """
    Constructs a list of Interval objects from 1D point data.

    Parameters
    ----------
    data : np.ndarray
        The sorted 2D array of (x, y) sample points.
    compute_normal : bool
        If True, computes the normal vector for each interval.

    Returns
    -------
    list[Interval]
        A list of connected Interval objects.
    """
    data = data[np.argsort(data[:, 0])]
    intervals = []
    for i in range(data.shape[0] - 1):
        interval = _Interval((i, i + 1), data)
        if i > 0:
            interval.neighbors[0] = intervals[-1]
            intervals[-1].neighbors[1] = interval
        if compute_normal:
            interval.compute_normal()
        intervals.append(interval)
    return intervals


def _build_triangles(
        data: np.ndarray,
        compute_normal: bool,
        delaunay: Delaunay | None,
        n_new_samples: int
) -> tuple[list[_Triangle], Delaunay]:
    """
    Constructs a list of Triangle objects using Delaunay triangulation.

    This function uses ``scipy.spatial.Delaunay`` to perform the triangulation.
    While there are various algorithms for Delaunay triangulation (e.g.,
    Bowyer-Watson, Fortune's sweep), scipy.spatial.Delaunay uses Qhull. The
    algorithm works by lifting the 2D points onto a 3D paraboloid, computing
    their convex hull, and projecting the lower hull faces back onto the 2D
    plane to obtain the Delaunay triangulation. For details, see:

    - Delaunay triangulation : https://en.wikipedia.org/wiki/Delaunay_triangulation
    - Triangulation theory : https://i.cs.hku.hk/~provinci/training11/delaunay.pdf
    - Bowyer-Watson : https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
    - Fortune's sweep : https://en.wikipedia.org/wiki/Fortune%27s_algorithm
    - Qhull : https://en.wikipedia.org/wiki/Quickhull

    Parameters
    ----------
    data : np.ndarray
        The 3D array of (x, y, z) sample points.
    compute_normal : bool
        If True, computes the normal vector for each triangle.
    delaunay : Delaunay or None
        An existing Delaunay object for incremental updates. If None, a new
        triangulation is created.
    n_new_samples : int
        The number of new points added to the data since the last update.

    Returns
    -------
    list[Triangle]
        A list of connected Triangle objects forming the Delaunay triangulation.
    Delaunay
        The updated or newly created Delaunay triangulation object used for
        incremental updates.
    """
    if delaunay is None:
        delaunay = Delaunay(data[:, :-1], incremental=True)
    else:
        delaunay.add_points(data[-n_new_samples:, :-1])
    triangles = []
    for point_idx in delaunay.simplices:
        triangle = _Triangle(tuple(point_idx), data)
        if compute_normal:
            triangle.compute_normal()
        triangles.append(triangle)
    for i, neighbors in enumerate(delaunay.neighbors):
        triangle = triangles[i]
        for j, idx in enumerate(neighbors):
            if idx == -1:
                continue
            triangle.neighbors[j] = triangles[idx]
    return triangles, delaunay


def _compute_badness(
        meshes: list[_Interval] | list[_Triangle],
        contour_levels: np.ndarray,
        contour_only: bool
) -> None:
    """
    Calculates and assigns a 'badness' metric to each mesh element.

    The badness is determined by two criteria:
    
    1. Curvature : The curvature between two adjacent mesh elements.
    2. Contour Intersection : A badness of inf is assigned if any contour level
        passes through the element. This ensures that mesh elements intersecting
        with contour lines are always refined.

    Parameters
    ----------
    meshes : list[Interval] or list[Triangle]
        The list of mesh elements to evaluate.
    contour_levels : np.ndarray
        Contour levels to refine around.
    contour_only : bool
        If True, only the contour intersection criterion is used.
    """
    for mesh in meshes:
        if not contour_only:
            for i, neighbor in enumerate(mesh.neighbors):
                if not np.isnan(mesh.badness[i]) or neighbor is None:
                    continue
                curvature = _compute_curvature(mesh, neighbor)
                j = neighbor.neighbor_idx(mesh)
                mesh.badness[i] = neighbor.badness[j] = curvature

        values = mesh.vertices()[:, -1]
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if ((contour_levels > (vmin - FLT_EPS)) &
            (contour_levels < (vmax + FLT_EPS))).any():
            mesh.badness[-1] = np.inf
        else:
            mesh.badness[-1] = -np.inf


def _compute_curvature(
        mesh: _Interval | _Triangle,
        neighbor: _Interval | _Triangle
) -> float:
    """
    Computes the curvature between two adjacent mesh elements.

    The curvature is approximated by measuring the change in normal vectors
    between adjacent mesh elements rather than computing the exact geometric
    curvature. For more details, see:

    - Curvature : https://en.wikipedia.org/wiki/Curvature
    - Relation b/w curvature and normal vector : https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_%28Calculus%29/Vector_Calculus/2%3A_Vector-Valued_Functions_and_Motion_in_Space/2.3%3A_Curvature_and_Normal_Vectors_of_a_Curve

    Parameters
    ----------
    mesh : Interval or Triangle
        The first mesh element.
    neighbor : Interval or Triangle
        The adjacent mesh element.

    Returns
    -------
    float
        The angle between the normal vectors in radians.
    """
    v, w = mesh.normal, neighbor.normal
    angle = np.arccos(min(np.dot(v, w), 1))
    return angle if not np.isnan(angle) else -np.inf


def _refine_mesh(
        func: Callable[[np.ndarray], np.ndarray] |
              Callable[[np.ndarray, np.ndarray], np.ndarray],
        data: np.ndarray,
        meshes: list[_Interval] | list[_Triangle],
        theta: float,
        total_area: float,
        eps: float
) -> tuple[np.ndarray, int]:
    """
    Refines the mesh by adding new sample points.

    For triangle meshes, refinement is performed by adding points at edge
    midpoints rather than triangle centroids. While using centroids would result
    in fewer new points (splitting each triangle into 3 vs 4), it can lead to
    numerical instability due to the creation of needle-shaped triangles with
    poor aspect ratios.

    To prevent the creation of excessively small mesh elements, refinement is
    skipped if the length (for univariate sample) or area (for bivariate
    samples) of a mesh element is less than ``eps`` times the total sampling
    range.

    Parameters
    ----------
    func : Callable
        The function to sample for the new points.
    data : np.ndarray
        The current array of sample points.
    meshes : list[Interval] or list[Triangle]
        The list of mesh elements.
    theta : float
        The badness threshold.

    Returns
    -------
    np.ndarray
        The updated data array with new points.
    int
        The number of new samples added.
    """
    coords = []
    for mesh in meshes:
        print(mesh.area)
        if mesh.area > total_area * eps and mesh.max_badness() > theta:
            coords.append(mesh.midpoints())
    if not coords:
        return data, 0
    coords = np.vstack(coords)
    values = np.atleast_2d(func(*coords.T)).T
    new_data = np.hstack((coords, values))
    n_new_samples = new_data.shape[0]
    data = np.vstack((data, new_data))
    return data, n_new_samples
