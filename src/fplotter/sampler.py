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

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pandas as pd

from .types import Univariate, Bivariate, Function

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
    data : ma.MaskedArray of shape (n_samples, dim)
        The actual sample data. Each row is a sample, and each column is a
        dimension.
    grid_shape : ndarray of shape (dim,) or None
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
            data: ma.MaskedArray,
            grid_shape: npt.NDArray[float] | None
    ):
        self.data = data
        self.n_samples: int = data.shape[0]
        self.dim: int = data.shape[1]
        self.grid_shape = grid_shape

    def set_mask(
            self,
            mask: npt.NDArray[bool]
    ) -> None:
        """
        Sets specific data points to NaN using a given boolean mask.

        Parameters
        ----------
        mask : ndarray of shape (n_samples,)
            A boolean array. Data points at locations where the mask is **True**
            will be masked to NaN.
        """
        self.data.mask[:, -1] = mask

    def reshape_as_grid(
            self,
            apply_mask: bool
    ) -> tuple[npt.NDArray[float], ...]:
        """
        Reshapes the flat data array into a grid.

        Parameters
        ----------
        apply_mask : bool
            If True, masked values in the data array are filled with NaN values. 
            If False, returns the raw underlying data array without applying any 
            masking.

        Returns
        -------
        tuple of ndarrays
            A tuple of ndarrays, where each array represents a dimension of the
            data reshaped as a grid. The return value contains a ``dim`` number
            of reshaped ndarrays, each with size matching the ``grid_shape``.

        Raises
        ------
        AssertionError
            If the ``grid_shape`` attribute is None.
        """
        assert self.grid_shape is not None, "Grid shape is not set."
        data = self.data.data if not apply_mask else self.data.filled(np.nan)
        grids = [
            data[:, i].reshape(self.grid_shape)
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
    x : ndarray
        The x-coordinates of the sample points.
    y : ndarray
        The y-coordinates of the sample points (values).
    grid_shape : int or None
        For 1D grids, this is equal to the length of ``x``. See
         the `` Sample.grid_shape `` parameter for more details.
    """

    def __init__(
            self,
            x: npt.NDArray[float],
            y: npt.NDArray[float],
            grid_shape: int | None
    ):
        grid_shape = np.array([grid_shape]) if grid_shape is not None else None
        super().__init__(ma.column_stack((x, y)), grid_shape)

    def reshape_as_grid(
            self,
            apply_mask: bool
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        A specialization of the parent class's ``reshape_as_grid()``.

        Since univariate samples are inherently one-dimensional, no actual
        reshaping is needed.

        Parameters
        ----------
        apply_mask : bool
            If True, masked values in the data array are filled with NaN values. 
            If False, returns the raw underlying data array without applying any 
            masking.

        Returns
        -------
        ndarray of shape (n_samples,)
            The x-coordinates of the sample points.
        ndarray of shape (n_samples,)
            The y-coordinates (function values) of the sample points.

        Raises
        ------
        AssertionError
            If the ``grid_shape`` attribute is None.
        """
        assert self.grid_shape is not None, "Grid shape is not set."
        data = self.data.data if not apply_mask else self.data.filled(np.nan)
        return *data.T,


class Sample3d(Sample):
    """
    A specialization of ``Sample`` class for bivariate function sampling.

    Parameters
    ----------
    x : ndarray
        The x-coordinates of the sample points.
    y : ndarray
        The y-coordinates of the sample points.
    z : ndarray
        The z-coordinates of the sample points (values).
    grid_shape : (int, int) or None
        If the data was sampled from a 2D grid, this stores the shape of that
        grid, e.g., (nx, ny). See ``Sample.grid_shape`` for more details.
    """

    def __init__(
            self,
            x: npt.NDArray[float],
            y: npt.NDArray[float],
            z: npt.NDArray[float],
            grid_shape: tuple[int, int] | None
    ):
        if grid_shape is None:
            super().__init__(ma.column_stack((x, y, z)), None)
        else:
            super().__init__(ma.column_stack((x, y, z)), np.array(grid_shape))


""" SAMPLING FUNCTIONS """


def sample_uniform(
        func: Function[float],
        n_samples: int | tuple[int, int],
        xbound: tuple[float, float],
        ybound: tuple[float, float] | None = None,
) -> Sample:
    """
    Samples a univariate or bivariate function at uniform intervals.

    This is a factory function that samples either a univariate or bivariate
    function depending on the provided arguments. It acts as a bivariate
    function sampler if ``ybound`` is provided; otherwise, it samples a
    univariate function.

    Parameters
    ----------
    func : Function of float
        The univariate (y=f(x)) or bivariate (z=f(x,y)) function to sample.
    n_samples : int or tuple[int, int]
        For a univariate ``func``, the number of samples (int). For a bivariate 
        ``func``, a tuple with the number of samples for each axis, (nx, ny). If
        an integer value is provided for a bivariate ``func``, the same number
        of samples will be used for both axes (i.e., ``n_samples`` points on
        both x and y axes).
    xbound : tuple[float, float]
        The range (start, end) of the x-axis to sample. The start value must be
        less than the end value.
    ybound : tuple[float, float] or None, optional (default: None)
        The range (start, end) of the y-axis to sample, for bivariate functions.
        The start value must be less than the end value. This parameter is 
        required when sampling bivariate ``func``; otherwise the ``func`` will
        be treated as univariate which may result in errors or unexpected 
        behavior.

    Returns
    -------
    Sample
        Returns a ``Sample2d`` object for a univariate function, or a
        ``Sample3d`` object for a bivariate function.
    """
    if ybound is None:
        return _sample_univariate(func, n_samples, xbound)
    else:
        if isinstance(n_samples, int):
            n_samples = (n_samples, n_samples)
        return _sample_bivariate(func, n_samples, xbound, ybound)


def _sample_univariate(
        func: Univariate[float],
        n_samples: int,
        xbound: tuple[float, float]
) -> Sample2d:
    """
    Univariate helper function for ``sample_uniform``.

    Parameters
    ----------
    func : Univariate of float
        The univariate function to sample (y = f(x)).
    n_samples : int
        The number of points to sample.
    xbound : (float, float)
        The range (start, end) of the x-axis to sample.

    Returns
    -------
    Sample2d
        A ``Sample2d`` object containing the sampled (x, y) data.
    """
    x = np.linspace(*xbound, n_samples)
    y = func(x)
    return Sample2d(x, y, n_samples)


def _sample_bivariate(
        func: Bivariate[float],
        n_samples: tuple[int, int],
        xbound: tuple[float, float],
        ybound: tuple[float, float]
) -> Sample3d:
    """
    Bivariate helper function for ``sample_uniform``.

    Parameters
    ----------
    func : Bivariate of float
        The bivariate function to sample (z = f(x, y)).
    n_samples : (int, int)
        The number of points to sample on the x and y axes, (nx, ny).
    xbound : (float, float)
        The range (start, end) of the x-axis to sample.
    ybound : (float, float)
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
