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

from collections.abc import Iterable
from functools import wraps
from idlelib.pyparse import trans
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def apply_style(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with plt.style.context(self.style):
            return method(self, *args, **kwargs)

    return wrapper


class Plotter:
    def __init__(
            self,
            figure_size: tuple[float, float] = (4.6, 3.45),
            dpi: int = 300,
            style: str = files("plotter") / "styles" / "scientific.mplstyle"
    ):
        self.figure_size = figure_size
        self.dpi = dpi
        self.style = style
        self.figure: plt.Figure | None = None
        self.axes: plt.Axes | None = None

        self._init_figure()

    @apply_style
    def line(
            self,
            x: Iterable[float],
            y: Iterable[float],
            alpha: float = 1,
            linecolor: str | tuple[float, float, float] | None = None,
            linewidth: float = 1,
            linestyle: str = "solid",
            marker: str | None = None,
            markersize: float = 5,
            markeredgecolor: str | tuple[float, float, float] | None = None,
            markeredgewidth: float = 0,
            markerfacecolor: str | tuple[float, float, float] | None = None,
            zorder: int | None = None,
            legend: str | None = None
    ) -> None:
        self.axes.plot(
            x, y, alpha=alpha, color=linecolor, linewidth=linewidth,
            linestyle=linestyle, marker=marker, markersize=markersize,
            markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
            markerfacecolor=markerfacecolor, zorder=zorder, label=legend
        )

    @apply_style
    def contour(
            self,
            X: Iterable[float],
            Y: Iterable[float],
            Z: Iterable[float],
            levels: int | Iterable[float] | None = None,
            alpha: float = 1,
            color_map: str = "viridis",
            linecolor: str | tuple[float, float, float] | None = None,
            linewidth: float = 1,
            linestyle: str = "solid",
            colorbar: bool = True,
            label: bool = True
    ) -> None:
        color_map = color_map if linecolor is None else None
        contour_set = self.axes.contour(
            X, Y, Z, levels=levels, colors=linecolor, alpha=alpha,
            cmap=color_map, linewidths=linewidth, linestyles=linestyle
        )
        if label:
            plt.clabel(contour_set, inline_spacing=20)
        if colorbar and not (color_map is None and linecolor is not None):
            # Fake plot to get colorbar
            _, ax = plt.subplots()
            tmp = ax.contourf(
                X, Y, Z, vmin=min(contour_set.levels),
                vmax=max(contour_set.levels), levels=50, cmap=color_map
            )
            plt.colorbar(
                tmp, ax=self.axes, ticks=contour_set.levels,
                boundaries=contour_set.levels
            )

    @apply_style
    def heatmap(
            self,
            X: Iterable[float],
            Y: Iterable[float],
            Z: Iterable[float],
            alpha: float = 1,
            color_map: str = "viridis",
            contour: bool = True,
            levels: int | Iterable[float] | None = None,
            linecolor: str | tuple[float, float, float] = "white",
            linewidth: float | None = 0.75,
            linestyle: str = "solid",
            colorbar: bool = True,
            label: bool = True
    ) -> None:
        contour_fill = self.axes.contourf(
            X, Y, Z, levels=100, alpha=alpha, cmap=color_map
        )
        if contour:
            contour_set = self.axes.contour(
                X, Y, Z, levels=levels, colors=linecolor, alpha=alpha,
                linewidths=linewidth, linestyles=linestyle
            )
        if colorbar:
            plt.colorbar(contour_fill, ax=self.axes)
        if contour and label:
            plt.clabel(contour_set, inline_spacing=20)

    @apply_style
    def show(self) -> None:
        self.figure.show()

    @apply_style
    def save(
            self,
            file_name: str | Path,
            transparent: bool = True,
            dpi: int | str = "figure",
            format_: str = "png"
    ) -> None:
        self.figure.savefig(
            Path(file_name), transparent=transparent, dpi=dpi, format=format_
        )

    @apply_style
    def clear(self) -> None:
        self.figure.clf()
        self.axes.cla()
        self._init_figure()

    @apply_style
    def title(
            self,
            title: str
    ) -> None:
        self.axes.set_title(title)

    @apply_style
    def legend(
            self,
            position: str = "best",
            n_cols: int = 1,
            title: str | None = None
    ) -> None:
        self.axes.legend(
            loc=position, ncols=n_cols, title=title, handlelength=1,
            handletextpad=0.5
        )

    @apply_style
    def axis_label(
            self,
            xlabel: str | None = None,
            ylabel: str | None = None
    ) -> None:
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)

    @apply_style
    def axis_scale(
            self,
            xscale: str = "linear",
            yscale: str = "linear"
    ) -> None:
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)

    @apply_style
    def axis_limit(
            self,
            xlimit: tuple[float, float] = None,
            ylimit: tuple[float, float] = None
    ) -> None:
        if xlimit is not None:
            self.axes.set_xlim(*xlimit)
        if ylimit is not None:
            self.axes.set_ylim(*ylimit)

    @apply_style
    def _init_figure(self) -> None:
        self.figure = plt.figure(figsize=self.figure_size, dpi=self.dpi,
                                 layout="tight")
        self.axes = self.figure.subplots()

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} at {hex(id(self))}> "
                f"size: {self.figure_size} dpi: {self.dpi}, style: {self.style}")

    __str__ = __repr__


def main():
    plotter = Plotter()

    x = np.linspace(-10, 10, 1000)
    y1 = np.sin(x) / x
    y2 = np.sin(x ** 2) / x

    plotter.line(x, y1, legend="$\\sin(x)/x$")
    plotter.line(x, y2, legend="$\\sin(x^2)/x$")
    plotter.title("Line Plot")
    plotter.legend()
    plotter.axis_label("$x$", "$f(x)$")
    plotter.show()
    plotter.clear()

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.sin(X) * Y + np.cos(Y) * X

    plotter.contour(X, Y, Z1)
    plotter.title("Contour Plot")
    plotter.axis_label("$x$", "$y$")
    plotter.show()
    plotter.clear()

    plotter.heatmap(X, Y, Z1)
    plotter.title("Heatmap Plot")
    plotter.axis_label("$x$", "$y$")
    plotter.show()
    plotter.save("test.png")
    plotter.clear()


if __name__ == '__main__':
    main()
