from typing import Any
from numbers import Integral, Real
from pathlib import Path
from functools import wraps, partial
from inspect import signature
from types import NoneType

import numpy.typing as npt

from validator.primitive import validate_bool, validate_float, \
    validate_nonnegative_float, \
    validate_str, validate_positive_int, validate_str_literal, validate_int
from validator.tuple import validate_float_tuple, validate_float_range
from validator.iterable import validate_and_process_float_iterable
from validator.path import validate_path, process_path

# TODO: Allow nan? inf?
# TODO: How to handle sample arg name but different validation?
# TODO: Add validation to Plotter.py

ALLOWED_LINESTYLES = {"solid", "dashed", "dashdot", "dotted"}
ALLOWED_LEGEND_POSITIONS = {
    "upper left", "upper right", "lower left", "lower right", "upper center",
    "lower center", "center left", "center right", "center", "best", "right"
}
ALLOWED_AXIS_SCALES = {"linear", "log", "symlog"}
ALLOWED_FORMATS = {"png", "pdf", "svg"}


def validate_figure_size(arg: Any, name: str) -> None:
    validate_float_tuple(arg, "figure_size", 2)
    if arg[0] <= 0 or arg[1] <= 0:
        raise ValueError(
            f"{name} must contain positive values, but got {arg}."
        )


def validate_and_process_data2d(
        x: Any,
        y: Any
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    x = validate_and_process_float_iterable(x, 'x', 1)
    y = validate_and_process_float_iterable(y, 'y', 1)
    if x.size != y.size:
        raise ValueError(
            f"x and y must have the same length, "
            f"but got lengths {x.size} and {y.size}."
        )
    if x.size < 2:
        raise ValueError(
            f"x and y must each have at least 2 elements, "
            f"but got {x.size} and {y.size}."
        )
    return x, y


def validate_and_process_data3d(
        X: Any,
        Y: Any,
        Z: Any
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    X = validate_and_process_float_iterable(X, 'X', 2)
    Y = validate_and_process_float_iterable(Y, 'Y', 2)
    Z = validate_and_process_float_iterable(Z, 'Z', 2)
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            f"X, Y, and Z must have the same shape, "
            f"but got {X.shape}, {Y.shape}, and {Z.shape}."
        )

    if X.size < 4:
        raise ValueError(
            f"X, Y, and Z must each have at least 4 elements, "
            f"but got {X.size}, {Y.size}, and {Z.size}."
        )
    return X, Y, Z


def validate_alpha(arg: Any, name: str) -> None:
    validate_float(arg, "alpha")
    if not 0 <= arg <= 1:
        raise ValueError(
            f"{name} must be within the interval [0, 1], but got {arg!r}."
        )


def validate_color(arg: Any, name: str) -> None:
    if not isinstance(arg, (str, tuple, NoneType)):
        raise TypeError(
            f"{name} must be a string, a tuple of 3 floats, or None "
            f"but got {type(arg).__name__}."
        )
    if isinstance(arg, tuple):
        if len(arg) != 3:
            raise TypeError(
                f"{name} must be a tuple of 3 floats, "
                f"but got length {len(arg)}."
            )
        if not all(isinstance(x, Real) for x in arg):
            raise TypeError(
                f"All elements of {name} must be floats, "
                f"but got {[type(x).__name__ for x in arg]}."
            )
        if not all(0 <= x <= 1 for x in arg):
            raise ValueError(
                f"Elements of {name} must be in [0, 1], but got {arg}."
            )


def validate_and_process_levels(levels: Any) -> int | npt.NDArray[float]:
    if isinstance(levels, Integral):
        return int(levels)
    levels = validate_and_process_float_iterable(levels, 'levels', 1)
    if levels.size == 0:
        raise ValueError("levels must contain at least 1 element, but got 0.")
    return levels


def validate_and_process_save_path(filename: Any, format_: Any) -> Path:
    validate_str(format_, "format_", True)
    filename = process_path(filename, "filename")
    validate_path(filename.parent, "parent directory of filename", True)
    ext = filename.suffix[1:] if filename.suffix else format_
    if ext is None:
        raise ValueError(
            "Unable to determine the file extension: both filename and format "
            "are missing or invalid."
        )
    if ext in ALLOWED_FORMATS:
        filename = filename.with_suffix(f".{ext}")
    else:
        raise ValueError(
            f"Unsupported format {ext!r}. "
            f"Supported formats are: {", ".join(ALLOWED_FORMATS)}."
        )
    return filename


VALIDATORS = {
    "figure_size": validate_figure_size,
    "figure_dpi": validate_positive_int,
    "style": partial(validate_path, exists=True),
    "data2d": validate_and_process_data2d,
    "data3d": validate_and_process_data3d,
    "alpha": validate_alpha,
    "linecolor": validate_color,
    "linewidth": validate_nonnegative_float,
    "linestyle": partial(validate_str_literal, allowed=ALLOWED_LINESTYLES),
    "marker": partial(validate_str, optional=True),
    "markersize": validate_nonnegative_float,
    "markeredgecolor": validate_color,
    "markeredgewidth": validate_nonnegative_float,
    "markerfacecolor": validate_color,
    "zorder": partial(validate_int, optional=True),
    "legend": partial(validate_str, optional=True),
    "levels": validate_and_process_levels,
    "colormap": validate_str,
    "colorbar": validate_bool,
    "label": validate_bool,
    "contour": validate_bool,
    "save_path": validate_and_process_save_path,
    "transparent": validate_bool,
    "save_dpi": partial(validate_positive_int, optional=True),
    "plot_title": validate_str,
    "position": partial(validate_str_literal, allowed=ALLOWED_LEGEND_POSITIONS),
    "n_cols": validate_positive_int,
    "legend_title": partial(validate_str, optional=True),
    "xlabel": partial(validate_str, optional=True),
    "ylabel": partial(validate_str, optional=True),
    "xscale": partial(validate_str_literal, allowed=ALLOWED_AXIS_SCALES),
    "yscale": partial(validate_str_literal, allowed=ALLOWED_AXIS_SCALES),
    "xlimit": partial(validate_float_range, optional=True),
    "ylimit": partial(validate_float_range, optional=True)
}

PREPROCESSING_KEYS = {"data2d", "data3d", "levels", "save_path"}


def validate(*keys: str):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            sig = signature(method)
            bind_args = sig.bind(self, *args, **kwargs)
            bind_args.apply_defaults()

            if "data2d" in keys:
                validator = VALIDATORS["data2d"]
                x, y = validator(
                    bind_args.arguments['x'], bind_args.arguments['y']
                )
                bind_args.arguments['x'] = x
                bind_args.arguments['y'] = y
            if "data3d" in keys:
                validator = VALIDATORS["data3d"]
                X, Y, Z = validator(
                    bind_args.arguments['X'], bind_args.arguments['Y'],
                    bind_args.arguments['Z']
                )
                bind_args.arguments['X'] = X
                bind_args.arguments['Y'] = Y
                bind_args.arguments['Z'] = Z
            if "levels" in keys:
                validator = VALIDATORS["levels"]
                levels = validator(bind_args.arguments['levels'])
                bind_args.arguments['levels'] = levels
            if "save_path" in keys:
                validator = VALIDATORS["save_path"]
                filename = validator(
                    bind_args.arguments['filename'],
                    bind_args.arguments['format_']
                )
                bind_args.arguments['filename'] = filename
                bind_args.arguments['format_'] = None

            for key in keys:
                if key in PREPROCESSING_KEYS:
                    continue
                validator = VALIDATORS[key]
                validator(bind_args.arguments[key], key)

            return method(*bind_args.args, **bind_args.kwargs)

        return wrapper

    return decorator
