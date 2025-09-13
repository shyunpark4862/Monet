import scipy.special as sp
import numpy as np

from fplotter import FunctionPlotter

plotter = FunctionPlotter()
plotter.function_heatmap(
    lambda x, y: np.sin(x) * y + np.cos(y) * x,
    (-5, 5), (-5, 5), zbound=(-2, 2), clip_line=True
)
plotter.show()
plotter.clear()