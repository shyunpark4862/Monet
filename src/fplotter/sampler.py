from collections.abc import Callable
from itertools import product
from abc import ABC, abstractmethod

import numpy as np

from sample import Sample, Sample3d, Sample2d


class Sampler(ABC):
    def __init__(
            self,
            func: Callable[[np.ndarray, ...], np.ndarray],
            bounds: np.ndarray,
            n_samples: np.ndarray | int
    ):
        self.func = func
        self.bounds = bounds
        self.n_samples = n_samples

    @abstractmethod
    def sample(self) -> Sample:
        pass


class UniformSampler(Sampler):
    def sample(self) -> Sample:
        grid_axis = []
        for bound, n in zip(self.bounds, self.n_samples):
            grid_axis.append(np.linspace(*bound, n))
        grids = np.meshgrid(*grid_axis, indexing="ij")
        values = self.func(*grids)
        coords = [g.ravel() for g in grids]
        data = np.column_stack((*coords, values.ravel()))
        return Sample(data, self.n_samples)


class UnivariateUniformSampler(UniformSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            n_samples: int
    ):
        super().__init__(func, np.atleast_2d(xbound), np.atleast_1d(n_samples))

    def sample(self) -> Sample2d:
        x = np.linspace(*self.bounds[0], self.n_samples[0])
        y = self.func(x)
        return Sample2d(x, y, self.n_samples)


class BivariateUniformSampler(UniformSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: tuple[int, int]
    ):
        super().__init__(func, np.array([xbound, ybound]),
                         np.atleast_1d(n_samples))

    def sample(self) -> Sample3d:
        x = np.linspace(*self.bounds[0], self.n_samples[0])
        y = np.linspace(*self.bounds[1], self.n_samples[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = self.func(X, Y)
        return Sample3d(X.ravel(), Y.ravel(), Z.ravel(), self.n_samples)


class RandomSampler(Sampler):
    def __init__(
            self,
            func: Callable[[np.ndarray, ...], np.ndarray],
            bounds: np.ndarray,
            n_samples: int,
            seed: int | None
    ):
        super().__init__(func, bounds, n_samples)
        np.random.seed(seed)

    def sample(self) -> Sample:
        dim = self.bounds.shape[0]
        lows, highs = self.bounds.T
        inner = np.random.uniform(lows, highs, (self.n_samples - 2 ** dim, dim))
        corner = [
            np.where(np.array(bits) == 0, lows, highs)
            for bits in product([0, 1], repeat=dim)
        ]
        coords = np.vstack((inner, corner))
        values = self.func(*coords.T)
        return Sample(np.column_stack((coords, values)), None)


class UnivariateRandomSampler(RandomSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            n_samples: int,
            seed: int | None
    ):
        super().__init__(func, np.atleast_2d(xbound), n_samples, seed)

    def sample(self) -> Sample2d:
        x = np.random.uniform(*self.bounds[0], self.n_samples - 2)
        x = np.concatenate((x, self.bounds[0]))
        y = self.func(x)
        return Sample2d(x, y, None)


class BivariateRandomSampler(RandomSampler):
    def __init__(
            self,
            func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            xbound: tuple[float, float],
            ybound: tuple[float, float],
            n_samples: int,
            seed: int | None
    ):
        super().__init__(func, np.array([xbound, ybound]), n_samples, seed)

    def sample(self) -> Sample3d:
        inner = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.n_samples - 4, 2)
        )
        corner = [
            [self.bounds[0, 0], self.bounds[1, 0]],
            [self.bounds[0, 0], self.bounds[1, 1]],
            [self.bounds[0, 1], self.bounds[1, 0]],
            [self.bounds[0, 1], self.bounds[1, 1]],
        ]
        x, y = np.vstack((inner, corner)).T
        z = self.func(x, y)
        return Sample3d(x, y, z, None)


def sample_uniform(
        func: Callable[[np.ndarray, ...], np.ndarray],
        bounds: np.ndarray,
        n_samples: np.ndarray
) -> Sample:
    return UniformSampler(func, bounds, n_samples).sample()


def sample_uniform_univariate(
        func: Callable[[np.ndarray], np.ndarray],
        xbound: tuple[float, float],
        n_samples: int
):
    return UnivariateUniformSampler(func, xbound, n_samples).sample()


def sample_uniform_bivariate(
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        xbound: tuple[float, float],
        ybound: tuple[float, float],
        n_samples: tuple[int, int]
):
    return BivariateUniformSampler(func, xbound, ybound, n_samples).sample()


def sample_random(
        func: Callable[[np.ndarray, ...], np.ndarray],
        bounds: np.ndarray,
        n_samples: int,
        seed: int | None
):
    return RandomSampler(func, bounds, n_samples, seed).sample()


def sample_random_univariate(
        func: Callable[[np.ndarray], np.ndarray],
        xbound: tuple[float, float],
        n_samples: int,
        seed: int | None
):
    return UnivariateRandomSampler(func, xbound, n_samples, seed).sample()


def sample_random_bivariate(
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        xbound: tuple[float, float],
        ybound: tuple[float, float],
        n_samples: int,
        seed: int | None
):
    return BivariateRandomSampler(func, xbound, ybound, n_samples,
                                  seed).sample()
