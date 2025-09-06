from collections.abc import Callable

import numpy as np


class UniformSampler:
    def __init__(
            self,
            func: Callable[[np.ndarray, ...], np.ndarray],
            bounds: np.ndarray,
            n_samples: tuple[int, ...]
    ):
        self.func = func
        self.bounds = bounds
        self.n_samples = n_samples

    def run(self) -> tuple[np.ndarray, ...]:
        eval_points = []
        for bound, n in zip(self.bounds, self.n_samples):
            eval_points.append(np.linspace(*bound, n))
        X = np.meshgrid(*eval_points)
        Y = self.func(*X)
        return (*X, Y)


class UnivariateUniformSampler(UniformSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            n_samples: int = 100
    ):
        super().__init__(func, np.atleast_2d(xbound), (n_samples,))

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(*self.bounds[0], self.n_samples[0])
        y = self.func(x)
        return x, y


class BivariateUniformSampler(UniformSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int] = (10, 10)
    ):
        super().__init__(func, np.vstack((xbound, ybound)), n_samples)

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(*self.bounds[0], self.n_samples[0])
        y = np.linspace(*self.bounds[1], self.n_samples[1])
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        return X, Y, Z


if __name__ == '__main__':
    def f(x):
        return x ** 2


    def g(x, y):
        return x ** 2 + y ** 2


    def h(x, y, z):
        return x ** 2 + y ** 2 + z ** 2


    # sampler = UnivariateUniformSampler(f, (-10, 10))
    # sample = sampler.run()
    # print(sample)

    sampler = UniformSampler(h, np.array([[-10, 10], [-10, 10], [-10, 10]]),
                             (2, 2, 2))
    sample = sampler.run()
    print(sample)
