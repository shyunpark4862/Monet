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

from textwrap import indent
from typing import Callable

import numpy as np
import pandas as pd

""" SAMPLE CONTAINERS """


class Sample:
    """
    A base container for n-dimensional sample data points.
    
    This class stores sample points from functions mapping R^(n-1) to R. For a 
    function f: R^(n-1) -> R, the first n-1 columns of the data matrix contain 
    the input vector x in R^n, while the last column contains the corresponding 
    function value y = f(x). The function values are integrated as the final 
    column rather than stored separately.

    For grid-based sampling (e.g., using ``np.meshgrid()``), the data should be 
    flattened before being passed to this class, with the original grid shape 
    provided separately via the ``grid_shape`` parameter. The grid shape is 
    stored separately because this class does not assume that sampling must be 
    done on a rectangular grid - it can handle arbitrary sampling patterns as 
    well.

    
    Parameters
    ----------
    data : np.ndarray of shape (n_samples, dim)
        The actual sample data. Each row is a sample, and each column is a
        dimension.
    grid_shape : np.ndarray or None
        If the data was sampled from a grid, this stores the shape of that grid.
        e.g., (n,) for a 1D grid, (n, m) for a 2D grid. If sampling was not done
        on a rectangular grid, this parameter should be None. For n-dimensional
        samples, grid_shape should be an (n-1)-dimensional array, and the
        product of its elements must equal ``n_samples``.

    Attributes
    ----------
    n_samples : int
        The total number of samples (number of rows in ``data``).
    dim : int
        The dimension of each sample (number of columns in ``data``).
        
    Notes
    -----
    Currently, the codomain of the sampled function f is assumed to be R. 
    However, this may be extended to higher dimensions in the future to support 
    parameterized curves.
    """

    def __init__(
            self,
            data: np.ndarray,
            grid_shape: np.ndarray | None
    ):
        self.data = data
        self.n_samples: int = data.shape[0]
        self.dim: int = data.shape[1]
        self.grid_shape = grid_shape

    def set_mask(
            self,
            mask: np.ndarray
    ) -> None:
        """
        Sets specific data points to NaN using a given boolean mask.
        
        Parameters
        ----------
        mask : np.ndarray of shape (n_samples,)
            A boolean array. Data points at locations where the mask is **True**
            will be set to NaN.

        Warning
        -------
        This method permanently modifies the original data by setting masked 
        values to NaN. The original values cannot be recovered after masking. If 
        you need to preserve the original data, create a copy of the ``Sample``
        object using the ``copy()`` method before applying the mask.
        """
        self.data[mask, -1] = np.nan

    def reshape_as_grid(self) -> tuple[np.ndarray, ...]:
        """
        Reshapes the flat data array into a grid.

        Returns
        -------
        tuple[np.ndarray, ...]
            A tuple of NumPy arrays, where each array represents a dimension
            of the data reshaped as a grid. The return value contains ``dim`` 
            number of reshaped np.ndarrays, each with size matching the
            ``grid_shape``.

        Raises
        ------
        AssertionError
            If the ``grid_shape`` attribute is None.
        """
        assert self.grid_shape is not None, "Grid shape is not set."
        grids = [
            self.data[:, i].reshape(self.grid_shape)
            for i in range(self.dim)
        ]
        return *grids,

    def copy(self) -> Sample:
        """
        Creates a deep copy of the Sample object.

        Returns
        -------
        Sample
            A new Sample object with a copy of the data and grid shape.
        """
        sample = Sample(self.data.copy(), None)
        if self.grid_shape is not None:
            sample.grid_shape = self.grid_shape.copy()
        return sample

    """ Debugging """

    def debug_print(self) -> str:
        data = pd.DataFrame(self.data)
        data_str = str(data.round(3))
        data_str = "\n".join(data_str.split('\n')[:-2])
        data_str = indent(data_str, " " * 4)
        lines = [
            f"<{self.__class__.__name__} at {hex(id(self))}>",
            f"  @dim     : {self.dim}",
            f"  @size    : {self.n_samples}",
            f"  @meshgrid: {self.grid_shape}",
            f"  @data    :\n{data_str}"
        ]
        return '\n'.join(lines)

    """ Magic Methods """

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} at {hex(id(self))}> "
                f"dim: {self.dim} size: {self.n_samples}")

    __str__ = __repr__


class Sample2d(Sample):
    """
    A specialization of ``Sample`` class for univariate function sampling.
    
    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the sample points.
    y : np.ndarray
        The y-coordinates of the sample points (values).
    grid_shape : int or None
        For 1D grids, this is equal to the length of ``x``. See
        ``Sample.grid_shape`` parameter for more details.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            grid_shape: int | None
    ):
        if grid_shape is None:
            super().__init__(np.column_stack((x, y)), None)
        else:
            super().__init__(np.column_stack((x, y)), np.atleast_1d(grid_shape))

    def reshape_as_grid(self) -> tuple[np.ndarray, np.ndarray]:
        if self.grid_shape is None:
            return self.data[:, 0], self.data[:, 1]
        return *self.data.T,


class Sample3d(Sample):
    """
    A specialization of ``Sample`` class for bivariate function sampling.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the sample points.
    y : np.ndarray
        The y-coordinates of the sample points.
    z : np.ndarray
        The z-coordinates of the sample points (values).
    grid_shape : tuple[int, int] or None
        If the data was sampled from a 2D grid, this stores the shape of that
        grid, e.g., ``(nx, ny)``. See ``Sample.grid_shape`` parameter for more
        details.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            grid_shape: tuple[int, int] | None
    ):
        if grid_shape is None:
            super().__init__(np.column_stack((x, y, z)), None)
        else:
            super().__init__(np.column_stack((x, y, z)), np.array(grid_shape))


""" SAMPLING FUNCTIONS """


def sample_uniform(
        func: Callable[[np.ndarray], np.ndarray] |
              Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_samples: int | tuple[int, int],
        xbound: tuple[float, float],
        ybound: tuple[float, float] | None = None,
) -> Sample2d | Sample3d:
    """
    Samples a univariate or bivariate function at uniform intervals.

    This is a factory function that samples either a univariate or bivariate
    function depending on the provided arguments. It acts as a bivariate
    function sampler if ``ybound`` is provided; otherwise, it samples a
    univariate function.

    Parameters
    ----------
    func : Callable
        The univariate (y=f(x)) or bivariate (z=f(x,y)) function to sample.
    n_samples : int or tuple[int, int]
        For a univariate function, the number of samples (int). For a bivariate 
        function, a tuple with the number of samples for each axis, (nx, ny).
    xbound : tuple[float, float]
        The range (start, end) of the x-axis to sample. The start value must be
        less than the end value.
    ybound : tuple[float, float] or None, optional (default: None)
        The range (start, end) of the y-axis to sample, for bivariate functions.
        The start value must be less than the end value. This parameter is 
        required when sampling bivariate functions, otherwise the function will
        be treated as univariate which may result in errors or unexpected 
        behavior.

    Returns
    -------
    Sample2d or Sample3d
        Returns a ``Sample2d`` object for a univariate function, or a
        ``Sample3d`` object for a bivariate function.
    """
    if ybound is None:
        return _sample_uniform_univariate(func, n_samples, xbound)
    else:
        return _sample_uniform_bivariate(func, n_samples, xbound, ybound)


def _sample_uniform_univariate(
        func: Callable[[np.ndarray], np.ndarray],
        n_samples: int,
        xbound: tuple[float, float]
) -> Sample2d:
    """
    Univariate helper function for ``sample_uniform``.

    Parameters
    ----------
    func : Callable[[np.ndarray], np.ndarray]
        The univariate function to sample (y = f(x)).
    n_samples : int
        The number of points to sample.
    xbound : tuple[float, float]
        The range (start, end) of the x-axis to sample.

    Returns
    -------
    Sample2d
        A ``Sample2d`` object containing the sampled (x, y) data.
    """
    x = np.linspace(*xbound, n_samples)
    y = func(x)
    return Sample2d(x, y, n_samples)


def _sample_uniform_bivariate(
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_samples: tuple[int, int],
        xbound: tuple[float, float],
        ybound: tuple[float, float]
) -> Sample3d:
    """
    Bivariate helper function for ``sample_uniform``.

    Parameters
    ----------
    func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        The bivariate function to sample (z = f(x, y)).
    n_samples : tuple[int, int]
        The number of points to sample on the x and y axes, (nx, ny).
    xbound : tuple[float, float]
        The range (start, end) of the x-axis to sample.
    ybound : tuple[float, float]
        The range (start, end) of the y-axis to sample.

    Returns
    -------
    Sample3d
        A ``Sample3d`` object containing the sampled (x, y, z) data.
    """
    x = np.linspace(*xbound, n_samples[0])
    y = np.linspace(*ybound, n_samples[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = func(X, Y)
    return Sample3d(X.ravel(), Y.ravel(), Z.ravel(), n_samples)
