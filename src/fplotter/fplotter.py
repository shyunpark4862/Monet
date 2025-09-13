from collections.abc import Callable
from importlib.resources import files

import numpy as np

from plotter import Plotter
from . import sampler, clipper, resampler, refiner


class FunctionPlotter(Plotter):
    def __init__(
            self,
            figure_size: tuple[float, float] = (4.6, 3.45),
            dpi: int = 300,
            style: str = files("plotter") / "styles" / "scientific.mplstyle"
    ):
        super().__init__(figure_size, dpi, style)

    @staticmethod
    def _sampler(
            func: Callable[[np.ndarray], np.ndarray] |
                  Callable[[np.ndarray, np.ndarray], np.ndarray],
            n_samples: int | tuple[int, int],
            bounds: np.ndarray,
            clip_bound: tuple[float, float] | None,
            auto_clip: bool,
            k: float,
            n_iters: int,
            theta: float
    ) -> tuple[sampler.Sample2d | sampler.Sample3d, tuple[float, float] | None]:
        sample = sampler.sample(func, n_samples, *bounds)
        mask = np.repeat(False, sample.n_samples)
        if auto_clip or clip_bound is not None:
            mask, clip_bound = clipper.clip(sample, clip_bound, k)
        if n_iters > 0:
            if clip_bound is None:
                sample = refiner.refine(
                    func, sample, None, False, theta, n_iters
                )
            else:
                sample = refiner.refine(
                    func, sample, np.array(clip_bound), False, theta, n_iters
                )
            sample = resampler.resample(sample)
            mask, _ = clipper.clip(sample, clip_bound, k)
        sample.set_mask(mask)
        return sample, clip_bound

    def function_line(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            n_samples: int = 1000,
            ybound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            k: float = 1.5,
            n_iters: int = 3,
            theta: float = 0.1745,
            **kwargs
    ) -> None:
        sample, ybound = self._sampler(
            func, n_samples, np.atleast_2d(xbound), ybound, auto_clip, k,
            n_iters, theta
        )
        super().line(*sample.reshape_as_grid(True), **kwargs)
        if ybound is not None:
            super().axis_limit(ylimit=ybound)

    def function_contour(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (100, 100),
            zbound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            k: float = 3,
            clip_line: bool = True,
            n_iters: int = 3,
            theta: float = 0.1745,
            **kwargs
    ) -> None:
        sample, zbound = self._sampler(
            func, n_samples, np.array((xbound, ybound)), zbound, auto_clip, k,
            n_iters, theta
        )
        if clip_line and zbound is not None:
            self._draw_clip_shadow(sample, zbound)
        super().contour(*sample.reshape_as_grid(True), **kwargs)
        if clip_line and zbound is not None:
            self._draw_clip_line(sample, zbound)

    def function_heatmap(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (100, 100),
            zbound: tuple[float, float] | None = None,
            auto_clip: bool = True,
            k: float = 3,
            clip_line: bool = True,
            n_iters: int = 3,
            theta: float = 0.1745,
            **kwargs
    ):
        sample, zbound = self._sampler(
            func, n_samples, np.array((xbound, ybound)), zbound, auto_clip, k,
            n_iters, theta
        )
        if clip_line and zbound is not None:
            self._draw_clip_shadow(sample, zbound)
        super().heatmap(*sample.reshape_as_grid(True), **kwargs)
        if clip_line and zbound is not None:
            self._draw_clip_line(sample, zbound)

    def _draw_clip_shadow(
            self,
            sample: sampler.Sample3d,
            zlimit: tuple[float, float]
    ) -> None:
        self.axes.contourf(
            *sample.reshape_as_grid(False), levels=[-np.inf, *zlimit, np.inf],
            colors=["lightgray", "white", "lightgray"], alpha=0.5
        )

    def _draw_clip_line(
            self,
            sample: sampler.Sample3d,
            zlimit: tuple[float, float]
    ) -> None:
        super().contour(
            *sample.reshape_as_grid(False), levels=zlimit, linecolor="red",
            colorbar=False, label=False
        )


def main():
    import scipy.special as sp

    plotter = FunctionPlotter()
    plotter.function_line(
        lambda x: sp.gamma(x), (-3, 3), adaptive_refine=True, n_samples=1000,
        n_iters=5
    )
    plotter.show()
    plotter.clear()

    # plotter.function_contour(
    #     lambda x, y: sp.beta(x, y),
    #     (-5, 5), (-5, 5), zbound=(-3, 3), n_iters=5, n_samples=(100, 100)
    # )
    # plotter.show()
    # plotter.clear()

    # plotter.function_heatmap(
    #     lambda x, y: np.sin(x) * y + np.cos(y) * x,
    #     (-5, 5), (-5, 5), zbound=(-3, 3), n_samples=(100, 100)
    # )
    # plotter.show()
    # plotter.clear()


if __name__ == '__main__':
    main()
