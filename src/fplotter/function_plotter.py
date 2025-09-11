from collections.abc import Callable

import numpy as np

from src.plotter.plotter import Plotter
from sample import Sample3d
from sampler import sample_uniform_univariate, sample_uniform_bivariate
from clipper import clip_univariate, clip_bivariate
from resampler import resample_bivariate, resample_univariate
from refiner import refine_bivariate, refine_univariate


class FunctionPlotter(Plotter):
    def __init__(
            self,
            figure_size: tuple[float, float] = (4.6, 3.45),
            dpi: int = 300,
            style: str = "scientific.mplstyle"
    ):
        super().__init__(figure_size, dpi, style)

    def function_line(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            n_samples: int = 1000,
            ybound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            clip_coefficient: float = 1.5,
            adaptive_refine: bool = True,
            n_iters: int = 3,
            amr_threshold: float = 0.1745,
            **kwargs
    ) -> None:
        sample = sample_uniform_univariate(func, xbound, n_samples)
        if auto_clip or ybound is not None:
            clipped, ylimit = clip_univariate(sample, ybound, clip_coefficient)
        if adaptive_refine:
            sample = refine_univariate(
                func, sample, list(ylimit), False, amr_threshold, n_iters
            )
            sample = resample_univariate(sample)
            clipped, _ = clip_univariate(sample, ybound, clip_coefficient)
        super().line(*clipped.reshape_as_grid(), **kwargs)
        if ylimit is not None:
            super().axis_limit(ylimit=ylimit)

    def function_contour(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (100, 100),
            zbound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            clip_coefficient: float = 3,
            clip_line: bool = True,
            adaptive_refine: bool = True,
            n_iters: int = 3,
            amr_threshold: float = 0.1745,
            **kwargs
    ):
        sample = sample_uniform_bivariate(func, xbound, ybound, n_samples)
        if auto_clip or zbound is not None:
            clipped, zlimit = clip_bivariate(sample, zbound, clip_coefficient)
        if adaptive_refine:
            sample = refine_bivariate(
                func, sample, list(zlimit), True, amr_threshold, n_iters
            )
            sample = resample_bivariate(sample, "linear")
            clipped, _ = clip_bivariate(sample, zbound, clip_coefficient)
        if clip_line and zlimit is not None:
            self._draw_clip_shadow(sample, zlimit)
        super().contour(*clipped.reshape_as_grid(), **kwargs)
        if clip_line and zlimit is not None:
            self._draw_clip_line(sample, zlimit)

    def function_heatmap(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (100, 100),
            zbound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            clip_coefficient: float = 3,
            clip_line: bool = True,
            adaptive_refine: bool = True,
            n_iters: int = 3,
            amr_threshold: float = 0.1745,
            **kwargs
    ):
        sample = sample_uniform_bivariate(func, xbound, ybound, n_samples)
        if auto_clip or zbound is not None:
            clipped, zlimit = clip_bivariate(sample, zbound, clip_coefficient)
        if adaptive_refine:
            sample = refine_bivariate(
                func, sample, list(zlimit), False, amr_threshold, n_iters
            )
            sample = resample_bivariate(sample, "linear")
            clipped, _ = clip_bivariate(sample, zbound, clip_coefficient)
        if clip_line and zlimit is not None:
            self._draw_clip_shadow(sample, zlimit)
        super().heatmap(*clipped.reshape_as_grid(), **kwargs)
        if clip_line and zlimit is not None:
            self._draw_clip_line(sample, zlimit)

    def _draw_clip_shadow(
            self,
            sample: Sample3d,
            zlimit: tuple[float, float]
    ) -> None:
        self.axes.contourf(
            *sample.reshape_as_grid(), levels=[-np.inf, *zlimit, np.inf],
            alpha=0.5, colors=["lightgray", "white", "lightgray"]
        )

    def _draw_clip_line(
            self,
            sample: Sample3d,
            zlimit: tuple[float, float]
    ) -> None:
        super().contour(
            *sample.reshape_as_grid(), linecolor="red", levels=zlimit,
            colorbar=False, label=False
        )


def main():
    import scipy.special as sp

    plotter = FunctionPlotter()
    # plotter.function_line(
    #     lambda x: sp.gamma(x), (-3, 3), adaptive_refine=True, n_samples=1000,
    #     n_iters=5
    # )
    # plotter.show()
    # plotter.clear()

    plotter.function_contour(
        lambda x, y: sp.beta(x, y),
        (-5, 5), (-5, 5), zbound=(-3, 3), n_iters=5, n_samples=(100, 100)
    )
    plotter.show()
    plotter.clear()

    # plotter.function_heatmap(
    #     lambda x, y: np.sin(x) * y + np.cos(y) * x,
    #     (-5, 5), (-5, 5), zbound=(-3, 3), n_samples=(100, 100)
    # )
    # plotter.show()
    # plotter.clear()


if __name__ == '__main__':
    main()
