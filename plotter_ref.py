from typing import *

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import MaxNLocator

import numpy as np


class Plotter:
    def __init__(self):
        pass

    def plot(self, xs: Iterable[float], ys: Iterable[float], **kwargs) \
            -> Tuple[plt.Figure, plt.Axes]:
        plt.style.use('scientific.mplstyle')
        plt.plot(xs, ys, **kwargs)
        return plt.gcf(), plt.gca()

    def scatter(self, xs: Iterable[float], ys: Iterable[float], **kwargs) \
            -> Tuple[plt.Figure, plt.Axes]:
        kwargs.setdefault('s', 9)
        kwargs.setdefault('linewidth', 0)
        plt.style.use('scientific.mplstyle')
        plt.scatter(xs, ys, **kwargs)
        return plt.gcf(), plt.gca()

    def triplot(self, xs: Iterable[float], ys: Iterable[float], **kwargs) \
            -> Tuple[plt.Figure, plt.Axes]:
        plt.style.use('scientific.mplstyle')
        triangles = tri.Triangulation(xs, ys)
        plt.triplot(triangles, **kwargs)
        return plt.gcf(), plt.gca()

    def tricontourf(self, xs: Iterable[float], ys: Iterable[float],
                    zs: Iterable[float], **kwargs) \
            -> Tuple[plt.Figure, plt.Axes]:
        kwargs.setdefault('cmap', 'inferno')
        plt.style.use('scientific.mplstyle')
        plt.tricontourf(xs, ys, zs, **kwargs)
        return plt.gcf(), plt.gca()

    def trisurf(self, xs: Iterable[float], ys: Iterable[float],
                zs: Iterable[float], contour: bool = True, **kwargs) \
            -> Tuple[plt.Figure, plt.Axes]:
        kwargs.setdefault('cmap', 'inferno')
        kwargs.setdefault('antialiased', True)
        plt.style.use('scientific.mplstyle')
        ax = plt.figure(constrained_layout=True).add_subplot(projection='3d')
        ax.plot_trisurf(xs, ys, zs, **kwargs)
        if contour:
            ax.tricontour(xs, ys, zs, offset=min(zs), cmap='inferno')
        return plt.gcf(), plt.gca()

    def tricontour(self, xs: Iterable[float], ys: Iterable[float],
                   zs: Iterable[float], **kwargs) \
            -> Tuple[plt.Figure, plt.Axes]:
        kwargs.setdefault('cmap', 'inferno')
        plt.style.use('scientific.mplstyle')
        plt.tricontour(xs, ys, zs, **kwargs)
        return plt.gcf(), plt.gca()


if __name__ == '__main__':
    import numpy as np

    xs = np.random.random(100)
    ys = np.random.random(100)

    Plotter().scatter(xs, ys)
    plt.show()
