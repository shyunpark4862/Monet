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

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import griddata

from .sampler import Sample2d, Sample3d, Sample


def resample(
        sample: Sample,
        method: Literal["linear", "cubic", "nearest"] = "linear",
) -> Sample:
    """
    Resamples univariate (2D) or bivariate (3D) sample data to a rectangular
    grid.

    This is a factory function that dispatches to the appropriate resampling
    helper based on the dimension of the input ``sample``.

    For univariate (2D) data, interpolation is not needed since sorting the 
    samples by their x-coordinates implicitly defines a rectangular grid in 1D. 
    However, for bivariate (3D) data, interpolation is necessary since sorting x 
    and y coordinates separately is insufficient - values must be interpolated 
    at all combinations of x,y coordinates to create a proper rectangular grid.

    Parameters
    ----------
    sample : Sample
        The input sample object to be resampled.
    method : {"linear", "cubic", "nearest"}, optional (default: "linear")
        The interpolation method to use for resampling. This parameter is only
        used for bivariate (3D) data:
        - linear: Uses linear interpolation between points
        - cubic: Uses Bezier polynomial interpolation
        - nearest: Interpolates using values from nearest points

    Returns
    -------
    Sample
        A new Sample object containing the resampled data.

    Raises
    ------
    AssertionError
        If the dimension of the input ``sample`` is not 2 or 3.

    Notes
    -----
    This function does not modify the input ``sample``, it only returns a new
    sample object.
    """
    data = sample.data.data
    if sample.dim == 2:
        return _resample_univariate(data, 1e-6)
    elif sample.dim == 3:
        return _resample_bivariate(data, method, 1e-6)
    else:
        assert False, "Invalid sample dimension"


def _resample_univariate(
        data: npt.NDArray[float],
        eps: float
) -> Sample2d:
    """
    Resamples univariate (2D) data to a rectangular grid.

    For univariate (2D) data, sorting points by their x-coordinates
    automatically creates a rectangular grid. Points that are too close together
    are removed since they provide little visual benefit in plotting. The 
    proximity threshold is determined by multiplying eps with the total x-axis 
    range.

    Parameters
    ----------
    data : ndarray of shape (n_samples, 2)
        A ndarray representing the (x, y) data points.
    eps : float
        A small float value used to determine unique x and y coordinates.

    Returns
    -------
    Sample2d
        A new ``Sample2d`` object containing the thinned data.
    """
    data = data[data[:, 0].argsort()]
    x, y = data.T
    tol = (x[-1] - x[0]) * eps
    x_keep, y_keep = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if x[i] - x_keep[-1] > tol:
            x_keep.append(x[i])
            y_keep.append(y[i])
    return Sample2d(np.array(x_keep), np.array(y_keep), len(x_keep))


def _resample_bivariate(
        data: npt.NDArray[float],
        method: Literal["linear", "cubic", "nearest"],
        eps: float
) -> Sample3d:
    """
    Resamples bivariate (3D) data to a rectangular grid.

    The function internally uses ``scipy.interpolate.griddata()`` for
    interpolation. Available interpolation methods are:
    - nearest : Interpolates using values from nearest points
    - linear : First performs Delaunay triangulation using Qhull on x, y
        coordinates, then applies barycentric interpolation within each triangle
    - cubic : First performs Delaunay triangulation using Qhull on x, y
        coordinates, then uses Clough-Tocher scheme which fits a piecewise cubic
        Bezier polynomial on each triangle

    For more details on the Clough-Tocher scheme, see:
    - ``griddata`` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html
    - Related paper : Alfeld, Peter. "A trivariate clough—tocher scheme for
        tetrahedral data." Computer Aided Geometric Design 1.2 (1984): 169-181.
    - Related paper : Farin, Gerald. "Triangular bernstein-bézier patches."
        Computer Aided Geometric Design 3.2 (1986): 83-127.

    Sampling points that are too close together on the x and y axes will be
    removed since they provide little benefit for plotting. The proximity
    threshold is determined by multiplying eps with the axis range.

    Parameters
    ----------
    data : ndarray of shape (n_samples, 3)
        A ndarray representing the (x, y, z) data points.
    method : {"linear", "cubic", "nearest"}
        The interpolation method passed to ``scipy.interpolate.griddata``.
    eps : float
        A small float value used to determine unique x and y coordinates.

    Returns
    -------
    Sample3d
        A new ``Sample3d`` object containing the data interpolated on a regular
        grid.
    """
    x, y, z = data.T
    x, y = _sort(x, eps), _sort(y, eps)
    grid_shape = (len(x), len(y))
    X, Y = np.meshgrid(x, y, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    z = griddata(data[:, :-1], z, np.column_stack((x, y)), method)
    return Sample3d(x, y, z, grid_shape)


def _sort(
        arr: npt.NDArray[float],
        eps: float
) -> npt.NDArray[float]:
    """
    Sorts a 1D array and removes duplicate or closely spaced values.

    Parameters
    ----------
    arr : ndarray of shape (n,)
        The 1D array to be sorted and thinned.
    eps : float
        A small float value used to calculate the tolerance for removing close
        values.

    Returns
    -------
    ndarray of shape (n,)
        A sorted 1D array with closely spaced values removed.
    """
    arr = np.sort(arr)
    tol = (arr[-1] - arr[0]) * eps
    keep = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] - keep[-1] > tol:
            keep.append(arr[i])
    return np.array(keep)
