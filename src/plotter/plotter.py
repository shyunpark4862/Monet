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
from importlib.resources import files
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt


def _apply_style(method):
    """
    Applies the style before calling the wrapped method.

    Parameters
    ----------
    method : callable
        The plotting method to be wrapped.

    Returns
    -------
    callable
        The wrapper function that applies the style context.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with plt.style.context(self.style):
            return method(self, *args, **kwargs)

    return wrapper


class Plotter:
    """
    A thin wrapper class around matplotlib that provides consistent styling and
    simplified parameter options. It handles commonly adjusted graphic settings
    automatically while still allowing direct access to figure and axes objects
    for advanced customization when needed.

    This class is designed to create single plots (overlapping plots are 
    possible) rather than multiple subplots in one figure. For layouts requiring
    multiple plots, it is recommended to create individual plots and combine 
    them using image editing software rather than using matplotlib's subplot 
    capabilities.

    This class deliberately focuses on 2D plotting capabilities, not
    implementing 3D surface plots due to matplotlib's inherent limitations with
    3D visualization. As matplotlib originated as a 2D visualization library
    with 3D capabilities (``mpl_toolkits.mplot3d``) added later, it uses a
    CPU-based Painter's Algorithm for polygon rendering rather than a dedicated
    3D graphics engine. This results in significantly slower performance
    compared to GPU-accelerated alternatives, particularly for complex 3D
    surfaces. While 3D surface plotting may be added in future versions if
    needed, the current implementation aligns with matplotlib's primary strength
    in 2D visualization.
    
    Parameters
    ----------
    figure_size : tuple[float, float], optional (default: (4.6, 3.45))
        The width and height of the figure in inches.
    dpi : int, optional (default: 300)
        The resolution of the figure in dots per inch.
    style : str, optional (default: "scientific.mplstyle")
        The path to the matplotlib style file to be used for plotting. Style 
        files (``*.mplstyle``) located in src/plotter/styles/ directory, and 
        users can add their own style files to this directory for immediate use. 
        The default style is derived and modified from SciencePlots 
        (https://github.com/garrettj403/SciencePlots.git).

    Attributes
    ----------
    figure : plt.Figure
        The matplotlib Figure object.
    axes : plt.Axes
        The matplotlib Axes object.
    """

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

    """ Public Methods """

    @_apply_style
    def line(
            self,
            x: Iterable[float],
            y: Iterable[float],
            alpha: float = 1,
            linecolor: str | tuple[float, float, float] | None = None,
            linewidth: float = 1,
            linestyle: Literal[
                "solid", "dashed", "dashdot", "dotted"
            ] = "solid",
            marker: str | None = None,
            markersize: float = 5,
            markeredgecolor: str | tuple[float, float, float] | None = None,
            markeredgewidth: float = 0,
            markerfacecolor: str | tuple[float, float, float] | None = None,
            zorder: int | None = None,
            legend: str | None = None
    ) -> None:
        """
        Plots a line.
        
        Parameters
        ----------
        x : Iterable[float]
            The x-coordinates of the data points.
        y : Iterable[float]
            The y-coordinates of the data points.
        alpha : float, optional (default: 1)
            The transparency of all plot elements (line, marker face, marker 
            edge). Value ranges from 0 (completely transparent) to 1 (completely
            opaque). Cannot be set independently for different elements.
        linecolor : str or tuple[float, float, float] or None, optional
                    (default: None)
            The color of the line. Can be a color name as string (e.g. "red") 
            or RGB values as a tuple of floats between 0 and 1. Transparency 
            should be set via the ``alpha`` parameter, not as a fourth tuple 
            value.
        linewidth : float, optional (default: 1)
            The width of the line.
        linestyle : {"solid", "dashed", "dashdot", "dotted"}, optional
                    (default: "solid")
            The style of the line.
        marker : str or None, optional (default: None)
            The marker style for the data points.
        markersize : float, optional (default: 5)
            The size of the markers.
        markeredgecolor : str or tuple[float, float, float] or None, optional
                          (default: None)
            The color of the marker edges. Can be a color name as string 
            (e.g. "red") or RGB values as a tuple of floats between 0 and 1. 
            Transparency should be set via the ``alpha`` parameter, not as a  
            fourth tuple value.
        markeredgewidth : float, optional (default: 0)
            The width of the marker edges.
        markerfacecolor : str or tuple[float, float, float] or None, optional
                          (default: None)
            The color of the marker face. Can be a color name as string 
            (e.g. "red") or RGB values as a tuple of floats between 0 and 1. 
            Transparency should be set via the ``alpha`` parameter, not as a 
            fourth tuple value.
        zorder : int or None, optional (default: None)
            The z-order for plotting. Higher values appear in front of elements
            with lower values.
        legend : str or None, optional (default: None)
            The label for the line in the legend. Note that this only registers
            the label - the legend itself must be displayed by calling the
            ``legend()`` method.
        """
        self.axes.plot(
            x, y, alpha=alpha, color=linecolor, linewidth=linewidth,
            linestyle=linestyle, marker=marker, markersize=markersize,
            markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
            markerfacecolor=markerfacecolor, zorder=zorder, label=legend
        )

    @_apply_style
    def contour(
            self,
            X: Iterable[float],
            Y: Iterable[float],
            Z: Iterable[float],
            levels: int | Iterable[float] = 10,
            alpha: float = 1,
            color_map: str = "viridis",
            linecolor: str | tuple[float, float, float] | None = None,
            linewidth: float = 1,
            linestyle: Literal[
                "solid", "dashed", "dashdot", "dotted"
            ] = "solid",
            colorbar: bool = True,
            label: bool = True,
            zorder: int | None = None
    ) -> None:
        """
        Creates a contour plot.

        This method uses ``plt.contour()`` to draw the contour lines. When a
        colorbar is requested, it additionally uses ``plt.contourf()`` (filled
        contour) to generate a smooth colorbar instead of the default discrete
        one. This dual plotting may cause slight misalignment between colorbar
        ticks and contour levels at the boundaries. For better performance,
        consider setting ``colorbar=False`` if not needed.

        The matplotlib's contour implementation internally uses the ContourPy
        library, which employs an algorithm similar to Marching Squares to
        generate contour lines. The Marching Squares algorithm determines how
        contour lines intersect each cell of a rectangular grid by examining
        the values at the four corners. For more details, see:

        - ContourPy documentation: https://contourpy.readthedocs.io/en/v1.3.3/
        - Marching Squares: https://en.wikipedia.org/wiki/Marching_squares
        - Historical CONREC method (previously used by matplotlib):
          https://paulbourke.net/papers/conrec/

        Parameters
        ----------
        X : Iterable[float]
            The x-coordinates of the grid.
        Y : Iterable[float]
            The y-coordinates of the grid.
        Z : Iterable[float]
            The height values over the grid.
        levels : int or Iterable[float], optional (default: 10)
            If int, uses ``mpl.ticker.MaxNLocator`` to automatically determine
            at most n+1 "nice" contour levels between ``Z``'s min and max
            values. If Iterable[float], uses the specified levels to draw
            contours.
        alpha : float, optional (default: 1)
            The transparency of the contour lines.
        color_map : str, optional (default: "viridis")
            The colormap for the contour lines. Ignored if ``linecolor`` is
            provided.
        linecolor : str or tuple[float, float, float] or None, optional
                    (default: None)
            The color of the contour lines. Can be a color name as string 
            (e.g. "red") or RGB values as a tuple of floats between 0 and 1. 
            Transparency should be set via the ``alpha`` parameter, not as a 
            fourth tuple value. When provided, overrides ``color_map``.
        linewidth : float, optional (default: 1)
            The width of the contour lines.
        linestyle : {"solid", "dashed", "dashdot", "dotted"}, optional
                    (default: "solid")
            The style of the contour lines.
        colorbar : bool, optional (default: True)
            Whether to show a colorbar.
        label : bool, optional (default: True)
            Whether to label the contour lines.
        """
        color_map = color_map if linecolor is None else None
        contour_set = self.axes.contour(
            X, Y, Z, levels=levels, colors=linecolor, alpha=alpha,
            cmap=color_map, linewidths=linewidth, linestyles=linestyle
        )
        if label:
            plt.clabel(contour_set, inline_spacing=20)
        if colorbar and not (color_map is None and linecolor is not None):
            # Create filled contour plot for smooth colorbar
            _, ax = plt.subplots()
            tmp = ax.contourf(
                X, Y, Z, vmin=min(contour_set.levels),
                vmax=max(contour_set.levels), levels=50, cmap=color_map
            )
            plt.colorbar(
                tmp, ax=self.axes, ticks=contour_set.levels,
                boundaries=contour_set.levels
            )

    @_apply_style
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
            linestyle: Literal[
                "solid", "dashed", "dashdot", "dotted"
            ] = "solid",
            colorbar: bool = True,
            label: bool = True
    ) -> None:
        """
        Creates a heatmap.

        This method uses ``plt.contourf()`` with many levels (100) instead of 
        ``plt.imshow()`` since the input grid (``X``, ``Y``) is not assumed to 
        be uniform. While this provides more flexibility in handling arbitrary 
        grids, it may have performance implications for large data.

        Parameters
        ----------
        X : Iterable[float]
            The x-coordinates of the grid.
        Y : Iterable[float]
            The y-coordinates of the grid.
        Z : Iterable[float]
            The values for the heatmap.
        alpha : float, optional (default: 1)
            The transparency of the heatmap.
        color_map : str, optional (default: "viridis")
            The colormap for the heatmap.
        contour : bool, optional (default: True)
            Whether to overlay contour lines.
        levels : int or Iterable[float] or None, optional (default: None)
            The number of contour levels or specific levels.
        linecolor : str or tuple[float, float, float], optional
                    (default: "white")
            The color of the contour lines. Can be a color name as string 
            (e.g. "red") or RGB values as a tuple of floats between 0 and 1. 
            Transparency should be set via the ``alpha`` parameter, not as a 
            fourth tuple value.
        linewidth : float, optional (default: 0.75)
            The width of the contour lines.
        linestyle : {"solid", "dashed", "dashdot", "dotted"}, optional
                    (default: "solid")
            The style of the contour lines.
        colorbar : bool, optional (default: True)
            Whether to show a colorbar.
        label : bool, optional (default: True)
            Whether to label the contour lines.

        Notes
        -----
        Future versions may implement automatic resampling to a uniform grid 
        when appropriate, allowing the use of the more efficient
        ``plt.imshow()`` method.
        """
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

    @_apply_style
    def show(self) -> None:
        """Displays the current figure.
        
        Note
        ----
        This function typically does not flush the figure buffer automatically.
        Drawing subsequent plots without calling ``clear()`` after ``show()``
        can lead to overlapping or corrupted plots. Unless there's a specific
        reason not to, always call ``clear()`` after ``show()``.
        
        Warning
        -------
        The underlying ``plt.Figure.show()`` behavior can vary depending on the
        matplotlib backend being used. Some backends **may flush the figure
        buffer** after rendering. Due to this unpredictable behavior, it is 
        recommended to complete all graphic operations before calling either 
        ``show()`` or ``save()``, followed by ``clear()``.
        """
        self.figure.show()

    @_apply_style
    def save(
            self,
            filename: str | Path,
            transparent: bool = True,
            dpi: int | Literal["figure"] = "figure",
            format_: str | None = None,
    ) -> None:
        """
        Saves the current figure to a file.

        Parameters
        ----------
        filename : str or Path
            The name of the file to save.
        transparent : bool, optional (default: True)
            Whether to save the figure with a transparent background.
        dpi : int or {"figure"}, optional (default: "figure")
            The resolution for saving the figure. If "figure", uses the same dpi
            value as the created figure.
        format_ : str or None, optional (default: None)
            The file format. If provided, uses this format. If None, infers the
            format from the ``file_name`` extension. If both are None, defaults
            to PNG format.

        Notes
        -----
        Unlike ``show()``, this method never flushes the figure buffer, ensuring 
        that multiple calls to ``save()`` will produce identical output files as 
        long as the figure content remains unchanged. However, this also means 
        that drawing subsequent plots without calling ``clear()`` after 
        ``save()`` can lead to overlapping or corrupted plots. Unless there is 
        a specific reason not to, always call ``clear()`` after ``save()``.
        """
        self.figure.savefig(
            Path(filename), transparent=transparent, dpi=dpi, format=format_
        )

    @_apply_style
    def clear(self) -> None:
        """Clears the current figure and axes, and re-initializes them."""
        self.figure.clf()
        self.axes.cla()
        self._init_figure()

    @_apply_style
    def title(
            self,
            title: str
    ) -> None:
        """
        Sets the title of the plot.

        Parameters
        ----------
        title : str
            The title text.
        """
        self.axes.set_title(title)

    @_apply_style
    def legend(
            self,
            position: Literal[
                "upper left", "upper right", "lower left", "lower right",
                "upper center", "lower center", "center left", "center right",
                "center", "best", "right"
            ] = "best",
            n_cols: int = 1,
            title: str | None = None
    ) -> None:
        """
        Adds a legend to the plot.

        Parameters
        ----------
        position : {"upper left", "upper right", "lower left", "lower right",
                    "upper center", "lower center", "center left",
                    "center right", "center", "best", "right"}, optional
                    (default: "best")
            The position of the legend. When set to "best", matplotlib will
            automatically place the legend in the location that minimally
            overlaps with the plot elements. While convenient, this automatic
            positioning can impact performance. For better performance, consider
            specifying a fixed position.
        n_cols : int, optional (default: 1)
            The number of columns in the legend. Legend items are arranged by
            filling horizontally first, then vertically.
        title : str or None, optional (default: None)
            The title for the legend.
        """
        self.axes.legend(
            loc=position, ncols=n_cols, title=title, handlelength=1,
            handletextpad=0.5
        )

    @_apply_style
    def axis_label(
            self,
            xlabel: str | None = None,
            ylabel: str | None = None
    ) -> None:
        """
        Sets the labels for the x and y axes.

        Parameters
        ----------
        xlabel : str or None, optional (default: None)
            The label for the x-axis.
        ylabel : str or None, optional (default: None)
            The label for the y-axis.
        """
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)

    @_apply_style
    def axis_scale(
            self,
            xscale: Literal["linear", "log", "symlog"] = "linear",
            yscale: Literal["linear", "log", "symlog"] = "linear"
    ) -> None:
        """
        Sets the scale of the x and y axes.

        Non-positive values will be automatically clipped from the plot when
        using logarithmic scale ("log").
        
        Parameters
        ----------
        xscale : {"linear", "log", "symlog"}, optional (default: "linear")
            The scale for the x-axis (e.g., "linear", "log").
        yscale : {"linear", "log", "symlog"}, optional (default: "linear")
            The scale for the y-axis (e.g., "linear", "log").

        Notes
        -----
        There is a known technical issue where minor ticks are not displayed
        when using "symlog" (symmetric log) scale. This is a limitation of the
        underlying matplotlib implementation.
        """
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)

    @_apply_style
    def axis_limit(
            self,
            xlimit: tuple[float, float] = None,
            ylimit: tuple[float, float] = None
    ) -> None:
        """
        Sets the limits for the x and y axes.

        Parameters
        ----------
        xlimit : tuple[float, float] or None, optional (default: None)
            The limits for the x-axis.
        ylimit : tuple[float, float] or None, optional (default: None)
            The limits for the y-axis.
        
        
        Notes
        -----
        Matplotlib's 3D stuff is a real pain... It has
        ``mpl_toolkits.mplot3d.axes3d.Axes3D.set_zlim()`` but guess what? It 
        doesn't even properly clip the surface! Points outside the limits still 
        show up like they own the place. So you gotta do it yourself by setting
        those pesky out-of-bounds points to NaN. Honestly, whoever thought this
        was acceptable behavior needs to rethink their life choices...
        """
        if xlimit is not None:
            self.axes.set_xlim(*xlimit)
        if ylimit is not None:
            self.axes.set_ylim(*ylimit)

    """ Private Methods """

    @_apply_style
    def _init_figure(self) -> None:
        """Initializes the figure and axes objects."""
        self.figure = plt.figure(figsize=self.figure_size, dpi=self.dpi,
                                 layout="tight")
        self.axes = self.figure.subplots()

    """ Magic Methods """

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} at {hex(id(self))}> "
            f"size: {self.figure_size} dpi: {self.dpi}, style: {self.style}"
        )

    __str__ = __repr__
