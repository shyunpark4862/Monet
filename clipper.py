from sample import SampleND, Sample2D, Sample3D


class Clipper:
    def __init__(
            self,
            sample: SampleND,
            bound: tuple[float, float]
    ):
        self.sample = sample
        self.bound = bound

    def run(self) -> SampleND:
        mask = ((self.sample.data[:, -1] < self.bound[0])
                | (self.sample.data[:, -1] > self.bound[1]))
        self.sample.mask = mask
        return self.sample


class UnivariateClipper(Clipper):
    def __init__(
            self,
            sample: Sample2D,
            ybound: tuple[float, float]
    ):
        super().__init__(sample, ybound)

    def run(self) -> Sample2D:
        return super().run()


class BivariateClipper(Clipper):
    def __init__(
            self,
            sample: Sample3D,
            zbound: tuple[float, float]
    ):
        super().__init__(sample, zbound)

    def run(self) -> Sample3D:
        return super().run()


if __name__ == '__main__':
    import numpy as np

    data = np.random.rand(10, 2)
    sample = Sample2D(data)

    print(sample.debug_print())
