from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing import TypeAlias

    import polars as pl

    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series

OnRowError = Literal["null", "raise"]
