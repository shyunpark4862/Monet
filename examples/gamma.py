import pathlib
from typing import Final
from functools import partial

import numpy as np
import scipy.special as sp

from fplotter import FPlotter

FIG_DIR: Final[pathlib.Path] = pathlib.Path(__file__).parent / "figures"


def plot_gamma():
    fp = FPlotter()
    fp.fline(sp.gamma, (-5, 5), 1000)
    for x in [-2, -1, 0]:
        fp.axes.axvline(x, color="grey", linestyle="dashed", linewidth=0.75)
    fp.title("Gamma Function")
    fp.show()


def plot_loggamma():
    fp = FPlotter()
    fp.fline(sp.gamma, (0, 5), 1000, legend="$\\Gamma(x)$")
    fp.fline(sp.loggamma, (0, 5), 1000, legend="$\\log\\Gamma(x)$")
    fp.title("Log Gamma Function")
    fp.legend()
    fp.show()


def plot_beta():
    fp = FPlotter()
    fp.fheatmap(sp.beta, (-2, 2), (-2, 2), (100, 100))
    fp.title("Beta Function")
    fp.show()


def plot_polygamma():
    fp = FPlotter()
    fp.flines([partial(sp.polygamma, n) for n in [0, 1, 2]], (-3, 3), 1000)
    for x in [-2, -1, 0]:
        fp.axes.axvline(x, color="grey", linestyle="dashed", linewidth=0.75)
    fp.title("Polygamma Function")
    # fp.legend()
    fp.show()


def plot_erf():
    fp = FPlotter()
    fp.fline(sp.erf, (-3, 3), 1000)
    fp.title("Error Function")
    fp.show()


def plot_dawson():
    fp = FPlotter()
    fp.fline(sp.dawsn, (-3, 3), 1000, auto_clip=False)
    fp.title("Dawson Function")
    fp.show()


def plot_fresnel():
    fp = FPlotter()
    fp.fline(lambda x: sp.fresnel(x)[0], (-3, 3), 1000, legend="$S(x)$",
             auto_clip=False)
    fp.fline(lambda x: sp.fresnel(x)[1], (-3, 3), 1000, legend="$C(x)$",
             auto_clip=False)
    fp.title("Fresnel Function")
    fp.show()


## Legendre_p acts weirdly.
def plot_legendre():
    fp = FPlotter()
    for n in [0, 1, 2, 3, 4, 5]:
        fp.fline(lambda x: sp.legendre_p(n, x)[0], (-1, 1), 1000,
                 legend=f"$P_{n}(x)$", auto_clip=False)
    fp.title("Legendre Polynomial")
    fp.legend(n_cols=2)
    fp.show()


# How to set m, n?
def plot_associated_legendre():
    fp = FPlotter()
    for n, m in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        fp.fline(lambda x: sp.assoc_legendre_p(n, m, x)[0], (-1, 1), 1000,
                 legend=f"$P_{n}^{m}(x)$", auto_clip=False)
    fp.title("Associated Legendre Polynomial")
    fp.legend(n_cols=2)
    fp.show()

def plot_spherical_legendre():
    fp = FPlotter()
    for n, m in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        fp.fline(lambda x: sp.sph_legendre_p(n, m, x)[0], (-np.pi, np.pi), 1000,
                 legend=f"$P_{n}^{m}(x)$", auto_clip=False)
    fp.title("Associated Legendre Polynomial")
    fp.legend(n_cols=2)
    fp.show()

if __name__ == "__main__":
    plot_spherical_legendre()
