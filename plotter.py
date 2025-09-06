from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(
            self,
            figure_size: tuple[float, float] | None = None,
            dpi: int | None = None,
            layout: str | None = "tight",
            n_rows: int = 1,
            n_cols: int = 1,
            share_x: bool = False,
            share_y: bool = False
    ):
        with plt.style.context("scientific.mplstyle"):
            self.figure: plt.Figure = plt.figure(
                figsize=figure_size, dpi=dpi, layout=layout
            )
            self.axes: np.ndarray = self.figure.subplots(
                n_rows, n_cols, sharex=share_x, sharey=share_y, squeeze=False
            )
        self.active_axis: plt.Axes = self.axes[0, 0]

    def line(
            self,
            x: Iterable[float],
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
            legend: str | None = None,
            facet: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(facet)
        axis = self._set_projection(axis, "rectilinear")
        axis.plot(
            x, ys, alpha=alpha, color=linecolor, linewidth=linewidth,
            linestyle=linestyle, marker=marker, markersize=markersize,
            markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
            markerfacecolor=markerfacecolor, zorder=zorder, label=legend
        )

    def scatter(
            self,
            x: Iterable[float],
            y: Iterable[float],
            marker: str | None = None,
            markersize: float = 5,
            alpha: float | None = None,
            markeredgecolor: str | tuple[float, float, float] | None = None,
            markeredgewidth: float | None = None,
            markerfacecolor: str | tuple[float, float, float] | None = None,
            zorder: int | None = None,
            legend: str | None = None,
            facet: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(facet)
        axis = self._set_projection(axis, "rectilinear")
        axis.scatter(
            x, y, marker=marker, s=markersize, alpha=alpha,
            edgecolor=markeredgecolor, linewidth=markeredgewidth,
            facecolor=markerfacecolor, zorder=zorder, label=legend
        )

    def contour(
            self,
            X: Iterable[float],
            Y: Iterable[float],
            Z: Iterable[float],
            levels: int | Iterable[float] | None = None,
            alpha: float | None = None,
            color_map: str | None = "inferno",
            linecolor: str | tuple[float, float, float] | None = None,
            linewidth: float | None = None,
            linestyle: str | None = None,
            colorbar: bool = True,
            label: bool = True,
            facet: tuple[int, int] | int | None = None
    ) -> None:
        color_map = color_map if linecolor is None else None
        axis = self._get_axis(facet)
        axis = self._set_projection(axis, "rectilinear")
        contour_set = axis.contour(
            X, Y, Z, levels=levels, colors=linecolor, alpha=alpha,
            cmap=color_map, linewidths=linewidth, linestyles=linestyle
        )
        with plt.style.context("scientific.mplstyle"):
            if label:
                plt.clabel(contour_set, inline_spacing=20)
            if colorbar and not (color_map is None and linecolor is not None):
                # Fake plot to get colorbar
                _, ax = plt.subplots()
                tmp = ax.contourf(
                    X, Y, Z, vmin=min(contour_set.levels),
                    vmax=max(contour_set.levels), levels=50, cmap=color_map
                )
                plt.colorbar(tmp, ax=axis, boundaries=contour_set.levels)

    def heatmap(
            self,
            X: Iterable[float],
            Y: Iterable[float],
            Z: Iterable[float],
            alpha: float | None = None,
            color_map: str | None = "inferno",
            contour: bool = True,
            levels: int | Iterable[float] | None = None,
            linecolor: str | tuple[float, float, float] | None = "white",
            linewidth: float | None = 0.75,
            linestyle: str | None = None,
            colorbar: bool = True,
            label: bool = True,
            facet: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(facet)
        axis = self._set_projection(axis, "rectilinear")
        contour_fill = axis.contourf(
            X, Y, Z, levels=100, alpha=alpha, cmap=color_map
        )
        if contour:
            contour_set = axis.contour(
                X, Y, Z, levels=levels, colors=linecolor, alpha=alpha,
                linewidths=linewidth, linestyles=linestyle
            )
        with plt.style.context("scientific.mplstyle"):
            if colorbar:
                plt.colorbar(contour_fill, ax=axis)
            if contour and label:
                plt.clabel(contour_set, inline_spacing=20)

    def surface(
            self,
            X: Iterable[float],
            Y: Iterable[float],
            Z: Iterable[float],
            alpha: float | None = None,
            color_map: str = "inferno",
            facecolor: str | tuple[float, float, float] | None = None,
            contour: str = "bottom",
            levels: int | Iterable[float] | None = None,
            linewidth: float | None = None,
            linestyle: str | None = None,
            colorbar: bool = True,
            facet: tuple[int, int] | int | None = None
    ) -> None:
        color_map = color_map if facecolor is None else None
        axis = self._get_axis(facet)
        axis = self._set_projection(axis, "3d")
        surf = axis.plot_surface(
            X, Y, Z, alpha=alpha, cmap=color_map, color=facecolor, rstride=1,
            cstride=1
        )
        if contour in ["top", "bottom"]:
            axis.contour(
                X, Y, Z, levels=levels, alpha=alpha, cmap=color_map,
                linewidths=linewidth, linestyles=linestyle,
                offset=np.nanmin(Z) if contour == "bottom" else np.nanmax(Z)
            )
        with plt.style.context("scientific.mplstyle"):
            if colorbar and not (color_map is None and facecolor is not None):
                plt.colorbar(surf, ax=axis)

    def _get_axis(
            self,
            facet: tuple[int, int] | int | None,
    ) -> plt.Axes:
        if isinstance(facet, int):
            self.active_axis = self.axes.ravel()[facet]
        elif isinstance(facet, tuple):
            self.active_axis = self.axes[*facet]
        return self.active_axis

    def _set_projection(self, axis: plt.Axes, projection: str) -> plt.Axes:
        if plt.Axes.name == projection:
            return axis
        subplotspec = axis.get_subplotspec()
        axis.remove()
        with plt.style.context("scientific.mplstyle"):
            self.active_axis = self.figure.add_subplot(subplotspec,
                                                       projection=projection)
        return self.active_axis

    def show(self) -> None:
        with plt.style.context("scientific.mplstyle"):
            self.figure.show()

    def title(
            self,
            title: str,
            facet: tuple[int, int] | int | None = None,
    ) -> None:
        axis = self._get_axis(facet)
        axis.set_title(title)

    # TODO : Exception when axis is 3d
    def legend(
            self,
            position: str = "best",
            labels: list[str] | None = None,
            n_cols: int = 1,
            title: str | None = None,
            facet: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(facet)
        assert axis.name != "3d"

        if position in ["best", "upper right", "upper left", "lower left",
                        "lower right", "right", "center left", "center right",
                        "lower center", "upper center", "center"]:
            loc = position
            bbox_to_anchor = None
            borderaxespad = None
        elif position == "outer bottom":
            loc = "upper center"
            bbox_to_anchor = (0.5, -0.15)
            borderaxespad = 0
            n_cols = len(axis.get_legend_handles_labels()[0])
        elif position == "outer right":
            loc = "center left"
            bbox_to_anchor = (1.05, 0.5)
            borderaxespad = 0
        else:
            raise ValueError("Unknown legend position.")

        with plt.style.context("scientific.mplstyle"):
            axis.legend(
                loc=loc, ncols=n_cols, labels=labels, title=title,
                handlelength=1, handletextpad=0.5,
                bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad
            )

    def axis_label(
            self,
            xlabel: str | None = None,
            ylabel: str | None = None,
            zlabel: str | None = None,
            facet: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(facet)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if axis.name == "3d":
            axis.set_zlabel(zlabel)

    # TODO: Log scale in 3d
    def axis_scale(
            self,
            xscale: str = "linear",
            ysclae: str = "linear",
            facet: tuple[int, int] | int | None = None
    ) -> None:
        axis = self._get_axis(facet)
        assert axis.name != "3d"
        axis.set_xscale(xscale)
        axis.set_yscale(ysclae)


if __name__ == '__main__':
    from sampler import UnivariateUniformSampler, BivariateUniformSampler


    def g(x):
        return np.sin(x)


    def f(x):
        return np.cos(x)


    def h(x):
        return np.sin(x) * np.cos(x)


    def func3d(x, y):
        return np.abs(np.sin(x) * y + np.cos(y) * x) + 1


    X1, Y1 = UnivariateUniformSampler(f, (-20 * np.pi, 20 * np.pi),
                                      n_samples=1000).run()
    # X2, Y2 = UnivariateUniformSampler(g, (-2 * np.pi, 2 * np.pi)).run()
    # X3, Y3 = UnivariateUniformSampler(h, (-2 * np.pi, 2 * np.pi)).run()
    #
    X4, Y4, Z4 = BivariateUniformSampler(func3d, (1, 2 * np.pi),
                                         (1, 2 * np.pi)).run()

    plotter = Plotter()
    # plotter.line(X1, Y1, legend="$\\sin(x)$")
    # plotter.line(X2, Y2, legend="$\\cos(x)$")
    # plotter.line(X3, Y3, legend="WTF")
    plotter.heatmap(X4, Y4, Z4, colorbar=False)
    plotter.axis_label("$x$", "$y$", "$z$")
    plotter.title("Plot of $\\sin(x)y + \\cos(y)x$")
    plotter.axis_scale("log", "log")
    plotter.show()
