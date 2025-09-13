import numpy as np

def univariate_func(x):
    return np.sin(x)


def bivariate_func(x, y):
    return np.sin(x) * y + np.cos(y) * x


def trivariate_func(x, y, z):
    return np.sin(x) * y + np.cos(y) * z + np.tan(z) * x
