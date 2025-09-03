from __future__ import annotations

from dataclasses import dataclass
from socket import send_fds
from textwrap import indent

import numpy as np
import pandas as pd


class Mesh:
    pass


@dataclass
class RectangleMesh(Mesh):
    shape: tuple[int, ...]
    sample: SampleND


class SampleND:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.size: int = data.shape[0]
        self.dim: int = data.shape[1]
        self.mask: np.ndarray = np.repeat(False, self.size)
        self.mesh: Mesh | None = None
        self._clip_contours: list[np.ndarray] | None = None

    def build_mesh(
            self,
            shape: tuple,
            type_: str = "rectangle"
    ) -> None:
        if type_ == "rectangle":
            self.mesh = RectangleMesh(shape, self)
        else:
            raise ValueError("Unknown mesh type.")

    def get_plot_points(self) -> tuple[np.ndarray, ...]:
        if self.mesh is None:
            raise ValueError("Mesh is not built.")

        if isinstance(self.mesh, RectangleMesh):
            data = self.data.copy()
            data[self.mask, -1] = np.nan
            plot_points = [x.reshape(self.mesh.shape) for x in data.T]
            return tuple(plot_points)
        else:
            raise ValueError("Unknown mesh type.")

    # @property
    # def clip_contours(self) -> list[np.ndarray]:
    #     return self._clip_contours
    #
    # @clip_contours.setter
    # def clip_contours(self, value: list[np.ndarray] | np.ndarray):
    #     self._clip_contours = value

    def debug_print(self) -> str:
        data = pd.DataFrame(self.data)
        data["masked"] = self.mask
        data_str = str(data.round(3))
        data_str = "\n".join(data_str.split('\n')[:-2])
        data_str = indent(data_str, " " * 4)
        lines = [
            f"<{self.__class__.__name__} at {hex(id(self))}>",
            f"  @dim : {self.dim}",
            f"  @size: {self.size}",
            f"  @data:\n{data_str}"
        ]
        # if self.clip_contours:
        #     lines.append(f"  @clip_contours: {len(self.clip_contours)} contour(s)")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} at {hex(id(self))}> "
                f"dim: {self.dim} size: {self.size}")

    __str__ = __repr__


class Sample2D(SampleND):
    def __init__(self, data: np.ndarray):
        super().__init__(data)
        self.mesh = RectangleMesh((self.size,), self)

    # @property
    # def clip_contour(self) -> np.ndarray:
    #     return self._clip_contours

    def get_plot_points(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.data.copy()
        data[self.mask, -1] = np.nan
        return tuple(data.T)

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} at {hex(id(self))}> "
                f"size: {self.size}")

    __str__ = __repr__


class Sample3D(SampleND):
    def __init__(self, data: np.ndarray):
        super().__init__(data)

    def get_plot_points(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mesh is None:
            raise ValueError("Mesh is not built.")

        if isinstance(self.mesh, RectangleMesh):
            data = self.data.copy()
            data[self.mask, -1] = np.nan
            return (data[:, 0].reshape(self.mesh.shape),
                    data[:, 1].reshape(self.mesh.shape),
                    data[:, 2].reshape(self.mesh.shape))
        else:
            raise ValueError("Unknown mesh type.")

    def __repr__(self) -> str:
        lines = [
            f"<{self.__class__.__name__} at {hex(id(self))}> ",
            f"size: {self.size}"
        ]
        return '\n'.join(lines)

    __str__ = __repr__


if __name__ == '__main__':
    data = np.random.rand(10, 2)
    sample = Sample2D(data)
    print(sample.get_plot_points())
