from typing import Any
from pathlib import Path


def validate_and_process_path(
        arg: Any,
        name: str,
        exists: bool = False,
        optional: bool = False
) -> Path | None:
    if optional and arg is None:
        return None
    arg = process_path(arg, name)
    validate_path(arg, name, exists)
    return arg


def validate_path(
        arg: Path,
        name: str,
        exists: bool = False
) -> None:
    if not exists and not arg.exists():
        raise ValueError(f"{name} does not exist: {arg!r}")


def process_path(
        arg: Any,
        name: str
) -> Path:
    if isinstance(arg, Path):
        return arg
    try:
        return Path(arg)
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} cannot be converted to a Path object, but got {arg!r}."
        )
