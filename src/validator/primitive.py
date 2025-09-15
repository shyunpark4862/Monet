from typing import Any
from types import UnionType

from numbers import Real, Integral

import numpy as np


def _check_type(
        arg: Any,
        name: str,
        type_: type | UnionType,
        expected: str,
        optional: bool
) -> None:
    if optional and arg is None:
        return
    if not isinstance(arg, type_):
        msg = f"{name} should be {expected}"
        if optional:
            msg += " or None"
        msg += f", but got {type(arg).__name__}."
        raise TypeError(msg)


def _raise_value_error(
        arg: Any,
        name: str,
        expected: str
) -> None:
    raise ValueError(f"{name} should be {expected}, but got {arg!r}.")


def validate_int(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    _check_type(arg, name, Integral, "an integer", optional)


def validate_nonnegative_int(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    validate_int(arg, name, optional)
    if arg is not None and arg < 0:
        _raise_value_error(arg, name, "non-negative integer")


def validate_positive_int(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    validate_int(arg, name, optional)
    if arg is not None and arg <= 0:
        _raise_value_error(arg, name, "positive integer")


def validate_float(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    _check_type(arg, name, Real, "a float", optional)


def validate_nonnegative_float(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    validate_float(arg, name, optional)
    if arg is not None and arg < 0:
        _raise_value_error(arg, name, "non-negative float")


def validate_str(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    _check_type(arg, name, str, "a string", optional)


def validate_str_literal(
        arg: Any,
        name: str,
        allowed: set[str],
        optional: bool = False
) -> None:
    validate_str(arg, name, optional)
    if arg is not None and arg not in allowed:
        _raise_value_error(arg, name, f"one of {", ".join(allowed)}")


def validate_bool(
        arg: Any,
        name: str,
        optional: bool = False
) -> None:
    _check_type(arg, name, bool | np.bool, "a boolean", optional)
