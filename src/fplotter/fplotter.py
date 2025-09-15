from collections.abc import Callable, Iterable
from importlib.resources import files

import numpy as np
import numpy.typing as npt

from plotter import Plotter
from .clipper import clip
from .refiner import refine
from .resampler import resample
from .sampler import Sample, Sample2d, Sample3d, sample_uniform
from .types import Univariate, Bivariate, Function


class FPlotter(Plotter):
    """
    A plotter for 1D and 2D functions with adaptive sampling and refinement.

    This class extends the base ``Plotter`` to provide high-level methods for
    visualizing mathematical functions. It incorporates a data processing
    pipeline that includes initial uniform sampling, adaptive refinement in
    areas of high curvature or near specified contours, automatic data clipping
    to focus on relevant ranges, and resampling to a regular grid suitable for
    plotting.

    Parameters
    ----------
    figure_size : (float, float), optional (default: (4.6, 3.45))
        The width and height of the figure in inches.
    dpi : int, optional (default: 300)
        The resolution of the figure in dots per inch.
    style : str, optional (default: "scientific.mplstyle")
        The path to the matplotlib style file to be used for plotting. Style
        files (``*.mplstyle``) located in src/plotter/styles/ directory, and
        users can add their own style files to this directory for immediate use.
        The default style is derived and modified from SciencePlots
        (https://github.com/garrettj403/SciencePlots.git).
    """

    def __init__(
            self,
            figure_size: tuple[float, float] = (4.6, 3.45),
            dpi: int = 300,
            style: str = files("plotter") / "styles" / "scientific.mplstyle"
    ):
        super().__init__(figure_size, dpi, style)

    """ Public Methods """

    def flines(
            self,
            funcs: Iterable[Univariate[float]] | Univariate[float],
            xbound: tuple[float, float],
            n_samples: int = 100,
            ybound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            k: float = 1.5,
            n_iters: int = 3,
            theta: float = 0.1745,
            legends: Iterable[str] | None = None,
            **kwargs
    ):
        """
        Plots a 2D line graph of univariate functions y = f(x).

        The function is evaluated using an adaptive sampling algorithm that
        adds more points in regions of high curvature. The y-axis data can be
        automatically clipped to a "focus zone" to handle functions with
        extreme outliers or asymptotes.

        Parameters
        ----------
        funcs : Univariate of float or Iterable of such
            The univariate function(s) to plot. Must be vectorized to handle
            ndarray inputs. When plotting multiple functions, it is recommended
            to pass them together through this parameter rather than making
            multiple calls to ``flines()``, as auto-clipping considers the
            combined y-range of all functions.
        xbound : (float, float)
            The (min, max) range of the x-axis to be plotted. The first element
            must be smaller than the second element.
        n_samples : int, optional (default: 100)
            The number of initial samples.
        ybound : (float, float) or None, optional (default: None)
            The (min, max) clipping range for the y-axis. If provided, the first
            element must be smaller than the second element. Note that when
            ``ybound`` is provided, auto-clipping will not be performed even if
            ``auto_clip`` is True.
        auto_clip : bool, optional (default: True)
            If True, automatically determines the y-axis clipping range based
            on the interquartile range (IQR) of the data.
        k : float, optional (default: 1.5)
            The IQR coefficient for automatic clipping. A larger value results
            in a wider range.
        n_iters : int, optional (default: 3)
            The number of adaptive refinement iterations. Higher values produce
            better quality plots but may significantly increase computation time
            beyond a certain point.
        theta : float, optional (default: 0.1745 which is approx. 10 degrees)
            The curvature threshold in radians for refinement. Smaller values
            lead to more refinement.
        legends : Iterable of str or None, optional (default: None)
            The labels for the functions in the legend.
        **kwargs
            Additional graphical arguments are passed to ``Plotter.line()`.
        """
        funcs = list(funcs) if isinstance(funcs, Iterable) else [funcs]
        legends = legends if legends is not None else [None] * len(funcs)
        samples, ybound = self._process_samples(
            funcs, n_samples, np.atleast_2d(xbound), ybound, auto_clip, k,
            n_iters, theta, False
        )
        for sample, legend in zip(samples, legends):
            super().line(*sample.reshape_as_grid(True), legend=legend, **kwargs)
        if ybound is not None:
            super().axis_limit(ylimit=ybound)
        self._recover_xaxis_limit(xbound)

    def fcontour(
            self,
            func: Bivariate[float],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (50, 50),
            zbound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            k: float = 3,
            clip_line: bool = True,
            n_iters: int = 3,
            theta: float = 0.1745,
            **kwargs
    ) -> None:
        """
        Plots a contour graph of a bivariate function z = f(x, y).

        The function is evaluated on an adaptive grid that is refined in areas
        of high curvature. The z-axis data can be clipped to a "focus zone" to
        improve visualization of functions with large variations.

        Parameters
        ----------
        func : Bivariate of float
            The bivariate function to plot. Must be vectorized to handle
            ndarray inputs.
        xbound : (float, float)
            The (min, max) range of the x-axis. The first element must be
            smaller than the second element.
        ybound : (float, float)
            The (min, max) range of the y-axis. The first element must be
            smaller than the second element.
        n_samples : (int, int), optional (default: (25, 25))
            The initial grid size (nx, ny).
        zbound : (float, float) or None, optional (default: None)
            The (min, max) clipping range for the z-axis. If provided, the first
            element must be smaller than the second element. Note that when 
            ``zbound`` is provided, auto-clipping will not be performed even if 
            ``auto_clip`` is True.
        auto_clip : bool, optional (default: True)
            If True, automatically determines the z-axis clipping range.
        k : float, optional (default: 3)
            The IQR coefficient for automatic clipping.
        clip_line : bool, optional (default: True)
            If True, draws a red contour line indicating the clipping boundary.
        n_iters : int, optional (default: 3)
            The number of adaptive refinement iterations. Higher values produce
            better quality plots but may significantly increase computation time
            beyond a certain point.
        theta : float, optional (default: 0.1745 which is approx. 10 degrees)
            The curvature threshold in radians for refinement.
        **kwargs
            Additional graphical arguments are passed to ``Plotter.contour()``.

        Notes
        -----
        Unlike ``fheatmap()``, this method only refines the mesh at contour line
        intersections. This makes it more computationally efficient but
        potentially less accurate in regions between contours.
        """
        samples, zbound = self._process_samples(
            [func], n_samples, np.array((xbound, ybound)), zbound, auto_clip, k,
            n_iters, theta, True
        )
        sample = samples[0]
        if clip_line and zbound is not None:
            self._draw_clip_shadow(sample, zbound)
        super().contour(*sample.reshape_as_grid(True), **kwargs)
        if clip_line and zbound is not None:
            self._draw_clip_line(sample, zbound)

    def fheatmap(
            self,
            func: Bivariate[float],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (50, 50),
            zbound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            k: float = 3,
            clip_line: bool = True,
            n_iters: int = 3,
            theta: float = 0.1745,
            **kwargs
    ):
        """
        Plots a heatmap of a bivariate function ``z = f(x, y)``.

        This method is similar to ``fcontour()`` but uses a heatmap for
        visualization. It employs the same adaptive sampling and clipping
        pipeline.

        Parameters
        ----------
        func : Bivariate of float
            The bivariate function to plot. Must be vectorized to handle numpy
            array inputs.
        xbound : (float, float)
            The (min, max) range of the x-axis. The first element must be
            smaller than the second element.
        ybound : (float, float)
            The (min, max) range of the y-axis. The first element must be
            smaller than the second element.
        n_samples : (int, int), optional (default: (25, 25))
            The initial grid size (nx, ny).
        zbound : (float, float) or None, optional (default: None)
            The (min, max) clipping range for the z-axis. If provided, the first
            element must be smaller than the second element. Note that when 
            ``zbound`` is provided, auto-clipping will not be performed even if 
            ``auto_clip`` is True.
        auto_clip : bool, optional (default: True)
            If True, automatically determines the z-axis clipping range.
        k : float, optional (default: 3)
            The IQR coefficient for automatic clipping.
        clip_line : bool, optional (default: True)
            If True, draws a red contour line indicating the clipping boundary.
        n_iters : int, optional (default: 3)
            The number of adaptive refinement iterations. Higher values produce
            better quality plots but may significantly increase computation time
            beyond a certain point.
        theta : float, optional (default: 0.1745 which is approx. 10 degrees)
            The curvature threshold in radians for refinement.
        **kwargs
            Additional graphical arguments are passed to ``Plotter.heatmap()``.
        """
        samples, zbound = self._process_samples(
            [func], n_samples, np.array((xbound, ybound)), zbound, auto_clip, k,
            n_iters, theta, False
        )
        sample = samples[0]
        if clip_line and zbound is not None:
            self._draw_clip_shadow(sample, zbound)
        super().heatmap(*sample.reshape_as_grid(True), **kwargs)
        if clip_line and zbound is not None:
            self._draw_clip_line(sample, zbound)

    """ PRIVATE METHODS """

    @staticmethod
    def _process_samples(
            funcs: list[Function[float]],
            n_samples: int | tuple[int, int],
            bounds: npt.NDArray[float],
            clip_bound: tuple[float, float] | None,
            auto_clip: bool,
            k: float,
            n_iters: int,
            theta: float,
            contour_only: bool
    ) -> tuple[list[Sample], tuple[float, float] | None]:
        """
        Processes function samples through a pipeline of sampling, clipping,
        refining, and resampling.

        The sample processing pipeline consists of four stages:
        1. Initial sampling : Extracts samples from a uniform rectangular mesh
        2. Clipping : Determines appropriate clipping bounds from initial
            function values if auto-clipping is enabled
        3. Adaptive refinement : Refines the mesh in areas of high curvature and
            near clipping boundaries, stopping when mesh becomes too fine
        4. Resampling : Interpolates the refined irregular mesh back to a
            rectangular grid and applies the clipping mask

        Parameters
        ----------
        funcs : list of Function of float
            The mathematical function to be sampled.
        n_samples : int or (int, int)
            The number of initial samples to generate.
        bounds : ndarray
            The boundaries for the initial sampling domain.
        clip_bound : (float, float) or None
            A user-defined (lower, upper) boundary for clipping data.
        auto_clip : bool
            If True, calculates the clipping boundary automatically using the
            IQR method.
        k : float
            The IQR coefficient used for automatic clipping.
        n_iters : int
            The number of adaptive refinement iterations to perform.
        theta : float
            The curvature threshold (in radians) for refinement.

        Returns
        -------
        Sample
            The final, processed Sample object ready for plotting.
        (float, float) or None
            The (lower, upper) clipping boundary that was used. Returns None
            only if either auto_clip is False and no user-defined clip_bound is
            provided, or if auto_clip is True but the automatically calculated
            clipping boundary is invalid.

        Notes
        -----
        There is a subtle issue with the clipping process: The clipping bounds
        are determined only once from the initial samples and are not updated
        after refinement. This means that if the initial sample size is too
        small, the clipping bounds might not accurately represent the function's
        behavior, potentially leading to suboptimal visualization. This effect
        is particularly noticeable when using very small initial sample sizes.
        """
        samples = []
        for func in funcs:
            samples.append(sample_uniform(func, n_samples, *bounds))
        if auto_clip or clip_bound is not None:
            # Merge all samples into a single array for clipping
            merged_data = np.vstack([sample.data.data for sample in samples])
            merged_sample = Sample(np.ma.MaskedArray(merged_data), None)
            _, clip_bound = clip(merged_sample, clip_bound, k)
        if n_iters > 0:
            contour_levels = None if clip_bound is None \
                else np.array(clip_bound)
            for i, func in enumerate(funcs):
                sample = refine(
                    func, samples[i], contour_levels, contour_only, theta,
                    n_iters
                )
                samples[i] = resample(sample)
        if clip_bound is not None:
            for sample in samples:
                mask, _ = clip(sample, clip_bound, k)
                sample.set_mask(mask)
        return samples, clip_bound

    def _recover_xaxis_limit(self, xbound: tuple[float, float]):
        """
        Adjusts and recovers the x-axis limits.

        Parameters
        ----------
        xbound : (float, float)
            A tuple containing the lower and upper bounds of the x-axis.
        """
        x0, x1 = xbound
        xrange = x1 - x0
        margin = self.axes.margins()[0]
        super().axis_limit(xlimit=(x0 - margin * xrange, x1 + margin * xrange))

    def _draw_clip_shadow(
            self,
            sample: Sample3d,
            zbound: tuple[float, float]
    ) -> None:
        """
        Draws a shaded region to indicate clipped areas.

        Parameters
        ----------
        sample : Sample3d
            The sample data for the plot.
        zbound : (float, float)
            The (lower, upper) boundary of the focus zone.
        """
        self.axes.contourf(
            *sample.reshape_as_grid(False), levels=[-np.inf, *zbound, np.inf],
            colors=["lightgray", "white", "lightgray"], alpha=0.5
        )

    def _draw_clip_line(
            self,
            sample: Sample3d,
            zbound: tuple[float, float]
    ) -> None:
        """
        Draws a contour line at the clipping boundary.

        Parameters
        ----------
        sample : Sample3d
            The sample data for the plot.
        zbound : (float, float)
            The (lower, upper) boundary of the focus zone.
        """
        super().contour(
            *sample.reshape_as_grid(False), levels=zbound, linecolor="red",
            colorbar=False, label=False
        )
