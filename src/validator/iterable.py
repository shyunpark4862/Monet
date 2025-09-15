from typing import Any

import numpy as np
import numpy.typing as npt


def validate_and_process_float_iterable(
        arg: Any,
        name: str,
        dim: int | None = None,
        shape: tuple[int, ...] | None = None,
        size: int | None = None,
        optional: bool = False
) -> npt.NDArray[float] | None:
    if optional and arg is None:
        return None
    arg = _process_iterable(arg, name, float, "floats")
    _validate_ndarray(arg, name, dim, shape, size, np.number, "float")
    return arg


def validate_and_process_int_iterable(
        arg: Any,
        name: str,
        dim: int | None = None,
        shape: tuple[int, ...] | None = None,
        size: int | None = None,
        optional: bool = False
) -> npt.NDArray[int] | None:
    if optional and arg is None:
        return None
    arg = _process_iterable(arg, name, int, "integers")
    _validate_ndarray(arg, name, dim, shape, size, np.integer, "integer")
    return arg


def _process_iterable(
        arg: Any,
        name: str,
        dtype: type,
        expected: str
) -> npt.NDArray[Any]:
    try:
        return np.asarray(arg, dtype=dtype)
    except (ValueError, TypeError):
        raise TypeError(
            f"{name} must be a ndarray of {expected}, "
            f"but got {type(arg).__name__}."
        )


def _validate_ndarray(
        arg: npt.NDArray[Any],
        name: str,
        dim: int | None,
        shape: tuple[int, ...] | None,
        size: int | None,
        dtype: type | None,
        expected: str | None
) -> None:
    if (dtype is not None and expected is not None and
            not np.issubdtype(arg.dtype, dtype)):
        raise TypeError(
            f"{name} must contain {expected} elements, "
            f"but got dtype {arg.dtype}."
        )
    if dim is not None and arg.ndim != dim:
        raise ValueError(
            f"{name} must be {dim}-dimensional, but got {arg.ndim} dimensions."
        )
    if shape is not None and arg.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, but got {arg.shape}."
        )
    if size is not None and arg.size != size:
        raise ValueError(f"{name} must have size {size}, but got {arg.size}.")
