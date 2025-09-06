from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from plotter import Plotter
from sampler import UnivariateUniformSampler


class FunctionPlotter(Plotter):
    def __init__(
            self,
            figure_size: tuple[float, float] | None = None,
            dpi: int | None = None,
            layout: str | None = "tight",
    ):
        super().__init__(figure_size, dpi, layout, 1, 1, False, False)

    def line(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float] | None = None,
            n_samples: int = 100,
            **kwargs
    ) -> None:
        sampler = UnivariateUniformSampler(func, xbound, n_samples)
        X, Y = sampler.run()
        super().line(X, Y, **kwargs)


if __name__ == '__main__':
    def f(x):
        return np.sin(x)


    with plt.style.context("scientific.mplstyle"):
        plotter = FunctionPlotter()
        plotter.line(f, (-2 * np.pi, 2 * np.pi), legend="sin$(x)$")
        plotter.legend()
        plotter.show()
