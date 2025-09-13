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
import pandas as pd


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
        e.g., ``(n,)`` for a 1D grid, ``(n, m)`` for a 2D grid. If sampling was
        not done on a rectangular grid, this parameter should be None. For
        n-dimensional samples, grid_shape should be an (n-1)-dimensional array,
        and the product of its elements must equal ``n_samples``.

    Attributes
    ----------
    n_samples : int
        The total number of samples (number of rows in ``data``).
    dim : int
        The dimension of each sample (number of columns in ``data``).
        
    Note
    ____
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

        Warning
        -------
        This method requires the ``grid_shape`` attribute to be set (not None).
        If the sample points were not generated from a rectangular grid, this
        method will raise an assertion error.
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
        super().__init__(np.column_stack((x, y)), np.atleast_1d(grid_shape))

    def reshape_as_grid(self) -> tuple[np.ndarray, np.ndarray]:
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
        grid, e.g., ``(n_x, n_y)``. See ``Sample.grid_shape`` parameter for more
        details.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            grid_shape: tuple[int, int] | None
    ):
        super().__init__(np.column_stack((x, y, z)), np.array(grid_shape))
