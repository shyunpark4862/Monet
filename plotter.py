from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(
            self,
            figure_size: tuple[float, float] | None = None,
            dip: int | None = None,
            layout: str | None = "tight",
            n_rows: int = 1,
            n_cols: int = 1,
            share_x: bool = False,
            share_y: bool = False
    ):
        with plt.style.context("scientific.mplstyle"):
            self.figure: plt.Figure = plt.figure(
                figsize=figure_size, dpi=dip, layout=layout
            )
            self.axes: np.ndarray = self.figure.subplots(
                n_rows, n_cols, sharex=share_x, sharey=share_y, squeeze=False
            )
        self.active_axis: plt.Axes = self.axes[0, 0]

    def line(
            self,
            xs: Iterable[float],
            ys: Iterable[float],
            alpha: float | None = None,
            linecolor: str | tuple[float, float, float] | None = None,
            linewidth: float | None = None,
            linestyle: str | None = None,
            marker: str | None = None,
            markersize: float = 5,
            markeredgecolor: str | tuple[float, float, float] | None = None,
            markeredgewidth: float | None = None,
            markerfacecolor: str | tuple[float, float, float] | None = None,
            zorder: int | None = None,
            label: str | None = None,
            axis: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(axis)
        axis.plot(
            xs, ys, alpha=alpha, color=linecolor, linewidth=linewidth,
            linestyle=linestyle, marker=marker, markersize=markersize,
            markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
            markerfacecolor=markerfacecolor, zorder=zorder, label=label
        )

    def scatter(
            self,
            xs: Iterable[float],
            ys: Iterable[float],
            marker: str | None = None,
            markersize: float = 5,
            alpha: float | None = None,
            markeredgecolor: str | tuple[float, float, float] | None = None,
            markeredgewidth: float | None = None,
            markerfacecolor: str | tuple[float, float, float] | None = None,
            zorder: int | None = None,
            label: str | None = None,
            axis: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(axis)
        axis.scatter(
            xs, ys, marker=marker, s=markersize, alpha=alpha,
            edgecolor=markeredgecolor, linewidth=markeredgewidth,
            facecolor=markerfacecolor, zorder=zorder, label=label
        )

    # TODO: Support monochrome contour
    def contour(
            self,
            xs: Iterable[float],
            ys: Iterable[float],
            zs: Iterable[float],
            levels: int | Iterable[float] | None = None,
            alpha: float | None = None,
            color_map: str = 'inferno',
            linewidth: float | None = None,
            linestyle: str | None = None,
            label: str | None = None,
            zorder: int | None = None,
            colorbar: bool = True,
            digits: bool = True,
            axis: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(axis)
        cs = axis.contour(
            xs, ys, zs, levels=levels, alpha=alpha, cmap=color_map,
            linewidths=linewidth, linestyles=linestyle, label=label,
            zorder=zorder
        )
        with plt.style.context("scientific.mplstyle"):
            if digits:
                plt.clabel(cs, fontsize=10, inline_spacing=30)
            if colorbar:
                # TODO: Make colorbar smoother
                # Fake plot to get colorbar
                _, ax = plt.subplots()
                tmp = ax.contourf(xs, ys, zs, cmap=color_map)
                plt.colorbar(tmp, ax=axis)

    def _get_axis(
            self,
            axis: tuple[int, int] | int | None
    ) -> plt.Axes:
        return self.active_axis if axis is None else self.axes[axis]

    def show(self) -> None:
        self.figure.show()


if __name__ == '__main__':
    from sampler import BivariateUniformSampler


    def f(x, y):
        return x ** 2 + y ** 2


    sample = BivariateUniformSampler(f, (-1, 1), (-1, 1), (100, 100)).run()
    plotter = Plotter()
    plotter.contour(*sample.get_plot_points())
    plotter.show()
