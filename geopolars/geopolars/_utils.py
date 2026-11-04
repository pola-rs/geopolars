from __future__ import annotations

from pathlib import Path

LIB = Path(__file__).parent


def validate_coincidence_tolerances(
    abs_tol: float | None, rel_tol: float | None
) -> None:
    for name, tolerance in (("abs_tol", abs_tol), ("rel_tol", rel_tol)):
        if tolerance is not None and not tolerance >= 0:
            msg = f"`{name}` must be non-negative and not NaN, got {tolerance!r}"
            raise ValueError(msg)
