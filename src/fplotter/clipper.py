import numpy as np

from sample import Sample, Sample2d, Sample3d


class Clipper:
    def __init__(
            self,
            sample: Sample,
            bound: tuple[float, float] | None,
            iqr_threshold: float
    ):
        self.sample = sample.copy()
        self.bound = bound
        self.iqr_threshold = iqr_threshold

    def clip(
            self
    ) -> tuple[Sample, tuple[float, float] | None]:
        if self.bound is None:
            self.bound = self._get_focus_zone()
        mask = self._get_mask()
        if mask.any():
            self.sample.set_mask(self._get_mask())
            return self.sample, self.bound
        else:
            return self.sample, None

    def _get_focus_zone(self) -> tuple[float, float]:
        y = self.sample.data[:, -1]
        q1, q3 = np.nanpercentile(y, (25, 75))
        iqr = q3 - q1
        return q1 - self.iqr_threshold * iqr, q3 + self.iqr_threshold * iqr

    def _get_mask(self) -> np.ndarray:
        y = self.sample.data[:, -1]
        mask = (y < self.bound[0]) | (y > self.bound[1])
        return mask


class UnivariateClipper(Clipper):
    def __init__(
            self,
            sample: Sample2d,
            bound: tuple[float, float] | None,
            iqr_threshold: float
    ):
        super().__init__(sample, bound, iqr_threshold)

    def _get_mask(self) -> np.ndarray:
        mask = super()._get_mask()
        n = len(mask)
        i = 0
        while i < n:
            if mask[i]:
                start = i
                while i < n and mask[i]:
                    i += 1
                end = i - 1
                length = end - start + 1
                if length >= 3:
                    mask[start] = False
                    mask[end] = False
            else:
                i += 1
        return mask


def clip(
        sample: Sample,
        bound: tuple[float, float] | None,
        iqr_threshold: float
) -> tuple[Sample, tuple[float, float] | None]:
    return Clipper(sample, bound, iqr_threshold).clip()


def clip_bivariate(
        sample: Sample3d,
        bound: tuple[float, float] | None,
        iqr_threshold: float
) -> tuple[Sample3d, tuple[float, float] | None]:
    return Clipper(sample, bound, iqr_threshold).clip()


def clip_univariate(
        sample: Sample2d,
        bound: tuple[float, float] | None,
        iqr_threshold: float
) -> tuple[Sample2d, tuple[float, float] | None]:
    return UnivariateClipper(sample, bound, iqr_threshold).clip()
