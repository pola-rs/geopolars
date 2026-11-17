from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing import TypeAlias

    import polars as pl

    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series

OnRowError = Literal["null", "raise"]
EarlyCoordCheck = Literal["off", "finite", "finite_and_empty"]
ShortLines = Literal["null", "keep", "empty"]
EmptyPolygonExterior = Literal["keep", "null", "error", "empty"]
NewRingOrientation = Literal["keep", "ccw", "cw"]
NewHoles = Literal["opposite", "same"]
