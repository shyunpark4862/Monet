import numpy as np
from plotter import Plotter


def test_plotter_line(assert_figure):
    x = np.linspace(-10, 10, 1000)
    y1 = np.sin(x) / x
    y2 = np.sin(x ** 2) / x

    plotter = Plotter()
    plotter.line(x, y1, legend="$\\sin(x)/x$")
    plotter.line(x, y2, legend="$\\sin(x^2)/x$")
    plotter.title("Line Plot")
    plotter.legend()
    plotter.axis_label("$x$", "$f(x)$")

    assert_figure(plotter.figure)


def test_plotter_contour(assert_figure):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * Y + np.cos(Y) * X

    plotter = Plotter()
    plotter.contour(X, Y, Z)
    plotter.title("Contour Plot")
    plotter.axis_label("$x$", "$y$")

    assert_figure(plotter.figure)


def test_plotter_heatmap(assert_figure):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * X + np.cos(Y) * Y

    plotter = Plotter()
    plotter.heatmap(X, Y, Z)
    plotter.title("Heatmap Plot")
    plotter.axis_label("$x$", "$y$")

    assert_figure(plotter.figure)
