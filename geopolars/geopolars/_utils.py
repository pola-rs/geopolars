from __future__ import annotations

from pathlib import Path
from typing import get_args

LIB = Path(__file__).parent


def validate_one_of(name: str, value: str, options: object) -> None:
    if value not in get_args(options):
        msg = f"`{name}` must be one of {list(get_args(options))}, got {value!r}"
        raise ValueError(msg)


def validate_coincidence_tolerances(
    abs_tol: float | None, rel_tol: float | None
) -> None:
    for name, tolerance in (("abs_tol", abs_tol), ("rel_tol", rel_tol)):
        if tolerance is not None and not tolerance >= 0:
            msg = f"`{name}` must be non-negative and not NaN, got {tolerance!r}"
            raise ValueError(msg)
