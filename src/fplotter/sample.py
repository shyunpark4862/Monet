from __future__ import annotations

from textwrap import indent

import numpy as np
import pandas as pd


class Sample:
    def __init__(
            self,
            data: np.ndarray,
            grid_shape: np.ndarray | None
    ):
        self.data = data
        self.size: int = data.shape[0]
        self.dim: int = data.shape[1]
        self.grid_shape: np.ndarray | None = grid_shape

    def set_mask(self, mask: np.ndarray) -> None:
        self.data[mask, -1] = np.nan

    def reshape_as_grid(self) -> tuple[np.ndarray, ...]:
        grids = [
            self.data[:, i].reshape(self.grid_shape)
            for i in range(self.dim)
        ]
        return *grids,

    def copy(self) -> Sample:
        sample = Sample(self.data.copy(), None)
        if self.grid_shape is not None:
            sample.grid_shape = self.grid_shape.copy()
        return sample

    def debug_print(self) -> str:
        data = pd.DataFrame(self.data)
        data_str = str(data.round(3))
        data_str = "\n".join(data_str.split('\n')[:-2])
        data_str = indent(data_str, " " * 4)
        lines = [
            f"<{self.__class__.__name__} at {hex(id(self))}>",
            f"  @dim     : {self.dim}",
            f"  @size    : {self.size}",
            f"  @meshgrid: {self.grid_shape}",
            f"  @data    :\n{data_str}"
        ]
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} at {hex(id(self))}> "
                f"dim: {self.dim} size: {self.size}")

    __str__ = __repr__


class Sample2d(Sample):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            grid_shape: int | None
    ):
        super().__init__(np.column_stack((x, y)), np.atleast_1d(grid_shape))


class Sample3d(Sample):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            grid_shape: tuple[int, int] | None
    ):
        super().__init__(np.column_stack((x, y, z)), np.array(grid_shape))


if __name__ == '__main__':
    x = np.random.rand(10)
    y = np.random.rand(10)
    z = np.random.rand(10)
    w = np.random.rand(10)
    sample = Sample2d(x, y, 10)
    print(sample.debug_print())
