import numpy as np
from scipy.interpolate import griddata

from sample import Sample, Sample2d, Sample3d


class Resampler:
    def __init__(
            self,
            sample: Sample,
            method: str
    ):
        self.sample = sample
        self.method = method

    def resample(self) -> Sample:
        grid_axis = [self._sort(axis) for axis in self.sample.data[:, :-1].T]
        grid_shape = np.array([len(axis) for axis in grid_axis])
        grids = np.meshgrid(*grid_axis, indexing="ij")
        grids_flatten = np.column_stack(tuple(g.ravel() for g in grids))
        value = griddata(
            self.sample.data[:, :-1], self.sample.data[:, -1], grids_flatten,
            self.method
        )
        return Sample(np.column_stack((grids_flatten, value)), grid_shape)

    @staticmethod
    def _sort(arr: np.ndarray) -> np.ndarray:
        arr = np.sort(arr)
        tol = (arr[-1] - arr[0]) * 1e-6
        keep = [arr[0]]
        for i in range(1, len(arr)):
            if arr[i] - keep[-1] > tol:
                keep.append(arr[i])
        return np.array(keep)


class UnivariateResampler(Resampler):
    def __init__(self, sample: Sample2d):
        super().__init__(sample, "linear")

    def resample(self) -> Sample2d:
        data = self.sample.data.copy()
        data = data[data[:, 0].argsort()]
        x, y = data.T
        tol = (x[-1] - x[0]) * 1e-6
        x_keep, y_keep = [x[0]], [y[0]]
        for i in range(1, len(x)):
            if x[i] - x_keep[-1] > tol:
                x_keep.append(x[i])
                y_keep.append(y[i])
        return Sample2d(np.array(x_keep), np.array(y_keep), len(x_keep))


class BivariateResampler(Resampler):
    def __init__(self, sample: Sample2d, method: str):
        super().__init__(sample, method)

    def resample(self) -> Sample3d:
        x, y, z = self.sample.data.T
        x, y = self._sort(x), self._sort(y)
        grid_shape = (len(x), len(y))
        X, Y = np.meshgrid(x, y, indexing="ij")
        x, y = X.ravel(), Y.ravel()
        z = griddata(
            self.sample.data[:, :-1], z, np.column_stack((x, y)), self.method
        )
        return Sample3d(x, y, z, grid_shape)


def resample(
        sample: Sample,
        method: str
) -> Sample:
    return Resampler(sample, method).resample()


def resample_univariate(sample: Sample2d) -> Sample2d:
    return UnivariateResampler(sample).resample()


def resample_bivariate(
        sample: Sample2d,
        method: str
) -> Sample3d:
    return BivariateResampler(sample, method).resample()


if __name__ == '__main__':
    from sampler import sample_random_univariate, _sample_uniform_bivariate, \
        sample_random_bivariate

    sample = sample_random_bivariate(lambda x, y: np.sin(x) * np.cos(y), (0, 1),
                                     (0, 1), 10, seed=10)
    print(sample.data)
    resampler = BivariateResampler(sample, "cubic")
    resampled = resampler.resample()
    print(resampled.data)
