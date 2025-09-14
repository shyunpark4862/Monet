import scipy.special as sp

from fplotter import FPlotter
from test_functions import bivariate_func


def test_plotter_line(assert_figure):
    plotter = FPlotter()
    plotter.fline(sp.gamma, (-3, 3))
    plotter.title("Function Line Plot")
    plotter.axis_label("$x$", "$y$")
    assert_figure(plotter.figure)


def test_plotter_contour(assert_figure):
    plotter = FPlotter()
    plotter.fcontour(bivariate_func, (-5, 5), (-5, 5))
    plotter.title("Function Contour Plot")
    plotter.axis_label("$x$", "$y$")
    assert_figure(plotter.figure)


def test_plotter_heatmap(assert_figure):
    plotter = FPlotter()
    plotter.fheatmap(bivariate_func, (-5, 5), (-5, 5), zbound=(-2, 2))
    plotter.title("Function Heatmap Plot")
    plotter.axis_label("$x$", "$y$")
    assert_figure(plotter.figure)
