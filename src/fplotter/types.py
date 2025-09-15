from collections.abc import Callable
from typing import TypeVar, Final

import numpy as np
import numpy.typing as npt

T = TypeVar('T')
Univariate = Callable[[npt.NDArray[T]], npt.NDArray[T]]
Bivariate = Callable[[npt.NDArray[T], npt.NDArray[T]], npt.NDArray[T]]
Function = Univariate[T] | Bivariate[T]

FLT_EPS: Final[float] = np.finfo(float).eps  # Floating point epsilon
