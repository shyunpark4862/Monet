from collections.abc import Callable

import numpy as np

from sample import SampleND, Sample2D, Sample3D


class UniformSampler:
    def __init__(
            self,
            func: Callable[[np.ndarray, ...], np.ndarray],
            bounds: np.ndarray,
            n_samples: tuple
    ):
        self.func = func
        self.bounds = bounds
        self.n_samples = n_samples

    def run(self) -> SampleND:
        eval_points = []
        for bound, n in zip(self.bounds, self.n_samples):
            eval_points.append(np.linspace(*bound, n))
        X = np.meshgrid(*eval_points)
        Y = self.func(*X)
        data = np.column_stack((*[x.ravel() for x in X], Y.ravel()))
        return SampleND(data)


class UnivariateUniformSampler(UniformSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            n_samples: int = 100
    ):
        super().__init__(func, np.atleast_2d(xbound), (n_samples,))

    def run(self) -> Sample2D:
        xs = np.linspace(*self.bounds[0], self.n_samples[0])
        ys = self.func(xs)
        data = np.column_stack((xs, ys))
        return Sample2D(data)


class BivariateUniformSampler(UniformSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (10, 10)
    ):
        super().__init__(func, np.vstack((xbound, ybound)), n_samples)

    def run(self) -> Sample3D:
        xs = np.linspace(*self.bounds[0], self.n_samples[0])
        ys = np.linspace(*self.bounds[1], self.n_samples[1])
        X, Y = np.meshgrid(xs, ys)
        Z = self.func(X, Y)
        data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        sample = Sample3D(data)
        sample.build_mesh(self.n_samples, "rectangle")
        return sample


if __name__ == '__main__':
    def f(x):
        return x ** 2


    def g(x, y):
        return x ** 2 + y ** 2


    # sampler = UnivariateUniformSampler(f, (-10, 10))
    # sample = sampler.run()
    # print(sample)

    sampler = BivariateUniformSampler(g, (-10, 10), (-10, 10))
    sample = sampler.run()
    print(sample.get_plot_points())
