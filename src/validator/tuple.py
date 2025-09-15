from typing import Any
from numbers import Real


def validate_float_tuple(
        arg: Any,
        name: str,
        length: int,
        optional: bool = False
) -> None:
    if optional and arg is None:
        return
    if not isinstance(arg, tuple):
        raise TypeError(
            f"{name} must be a tuple of {length} floats, "
            f"but got {type(arg).__name__}."
        )
    if len(arg) != length:
        raise ValueError(
            f"{name} must be a tuple of {length} floats, "
            f"but got length {len(arg)}."
        )
    if not all(isinstance(x, Real) for x in arg):
        raise TypeError(
            f"All elements of {name} must be floats, "
            f"but got {[type(x).__name__ for x in arg]}."
        )


def validate_float_range(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    validate_float_tuple(arg, name, 2, optional)
    if arg is not None and arg[0] > arg[1]:
        raise ValueError(
            f"{name} must be a tuple (min, max) with min <= max, "
            f"but got ({arg[0]}, {arg[1]})."
        )
