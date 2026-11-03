"""Joining linestrings end to start into one (A-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn

Gaps = Literal["null", "bridge", "split"]


@doc(output="result")
def join_line_strings(
    line_strings: IntoExprColumn,
    *,
    gaps: Gaps = "null",
    drop_repeated_joins: bool = False,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    unique_z: bool = True,
    unique_m: bool = False,
) -> pl.Expr:
    """Joins the linestrings of each row into one, each one's end to the next one's start.

    | in | `gaps="null"`, `"bridge"` | `gaps="split"` |
    | - | - | - |
    | `list[LineStringXY[ZM]]` | `LineStringXY[ZM]` | `MultiLineStringXY[ZM]` |
    | `MultiLineStringXY[ZM]` | `LineStringXY[ZM]` | `MultiLineStringXY[ZM]` |
    | anything else | refused | refused |

    {keeps_dimension_and_crs}

    At each join, the last vertex of a linestring is compared with the first vertex of the next.
    Empty linestrings have no vertices to join, and are skipped.
    Where the two do not coincide, there is a gap.
    What is done with a gap is determined by `gaps`:

    | `gaps` | a row with a gap |
    | - | - |
    | `"null"` | becomes null |
    | `"bridge"` | is joined anyway, with a straight segment across the gap |
    | `"split"` | starts a new part at the gap |

    With `"split"`, the linestrings between two gaps are joined into one part.

    Both vertices at a join are kept, as they are in the input.
    With `drop_repeated_joins=True`, the first vertex of the next linestring is dropped where it coincides,
    so only the end of the linestring before it is kept.
    A gap keeps both its vertices either way.

    {coincidence_axes}
    {coincidence_tolerance}

    {empty_and_null}
    A null linestring or a null vertex in it also gives a null result.
    An empty list, or one of only empty linestrings, counts as empty.

    ```python
    df.select(geo.join_line_strings("legs", gaps="bridge"))
    ```
    """
    if gaps not in get_args(Gaps):
        msg = f"`gaps` must be one of {list(get_args(Gaps))}, got {gaps!r}"
        raise ValueError(msg)

    return register_plugin_function(
        plugin_path=LIB,
        args=[line_strings],
        function_name="join_line_strings",
        is_elementwise=True,
        kwargs={
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "unique_z": unique_z,
            "unique_m": unique_m,
            "gaps": gaps,
            "drop_repeated_joins": drop_repeated_joins,
        },
    )
