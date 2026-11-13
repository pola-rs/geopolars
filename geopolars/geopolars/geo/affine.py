"""Moving a geometry through space."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, PointType, PolygonType
from geopolars.geo._dispatch import on_geometry, on_geometry_pair
from geopolars.geo.centroid import _centroid
from geopolars.geo.construct.point import point

# TODO: maybe merge 'rotate' and 'rotate_about_centroid'?
# also the other ones.
# This will make the API easier to read I suppose,
# but is not consistent with Polars.

if TYPE_CHECKING:
    from typing import TypeAlias

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

    Offset: TypeAlias = (
        tuple[float | IntoExprColumn, float | IntoExprColumn]
        | tuple[float | IntoExprColumn, float | IntoExprColumn, float | IntoExprColumn]
    )

# fmt: off
_IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
]
# fmt: on


def _affine(column: pl.Expr, matrix: Sequence[float]) -> pl.Expr:
    """Transform every position (`x`, `y`, `z`) of a geometry by an affine matrix,
    given row by row as twelve numbers (see `AffineMatrix` in geopolars-compute).
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="affine",
        is_elementwise=True,
        kwargs={"coefficients": [float(c) for c in matrix]},
    )


def _cannot_translate_by_z(geometry: GeoArrowType) -> TypeError:
    return TypeError(
        f"cannot translate by a z offset: {geometry!r} has no z coordinate"
    )


def _translate(
    dx: float, dy: float, dz: float
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # Without this, translating an XY column by dz would quietly do nothing.
        if dz != 0.0 and "z" not in geometry._dimension:
            raise _cannot_translate_by_z(geometry)
        # fmt: off
        return _affine(column, (
            1, 0, 0, dx,
            0, 1, 0, dy,
            0, 0, 1, dz,
        ))
        # fmt: on

    return build


def _translate_by(
    column: pl.Expr, geometry: GeoArrowType, by: pl.Expr, offsets: GeoArrowType
) -> pl.Expr:
    _points("offset", offsets)
    # Whether a `z` in a column is 0 everywhere is only known once it is computed.
    if "z" in offsets._dimension and "z" not in geometry._dimension:
        raise _cannot_translate_by_z(geometry)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column, by],
        function_name="affine_then_shift",
        is_elementwise=True,
        kwargs={"coefficients": _IDENTITY},
    )


def translate(geometry: IntoExprColumn, offset: Offset | IntoExprColumn) -> pl.Expr:
    """Shift every coordinate of a geometry by an offset.

    Works on any geometry, of any dimension. An `m` value is a measure,
    not a position, so it is always carried through untouched.

    `offset` is either
    - `(dx, dy)` or `(dx, dy, dz)`, each one number for the whole column
      or a column (or expression) of numbers, one for every geometry, or
    - a `geoarrow.point` column (or a single point) with an offset for every geometry,
      its `x`, `y` and `z` being `dx`, `dy` and `dz`.

    ```python
    df.select(geo.translate("shape", (1.0, -2.0)))
    df.select(geo.translate("shape", ("dx", "dy", 0.5)))
    df.select(geo.translate("shape", "offset"))
    ```

    A geometry with a null offset comes out null.
    A `dz` is rejected for a geometry with no `z` rather than silently ignored:
    a number when it is not 0, and a column or a point with a `z` always,
    as whether that is 0 everywhere is not known in advance.
    A point column that declares a CRS has to declare the geometry's.

    This does not check if the new position might have wrapped around the antimeridian (± 180°).
    If your data might have, it would be best to call `wrap_longitude` on it as well.
    """
    if isinstance(offset, str | pl.Expr | pl.Series):
        return on_geometry_pair(geometry, offset, _translate_by)
    if len(offset) not in (2, 3) or not all(
        math.isfinite(d) for d in offset if isinstance(d, int | float)
    ):
        msg = (
            "`offset` has to be a point column, or two or three finite numbers "
            f"or columns, got: {offset!r}"
        )
        raise ValueError(msg)

    dx, dy, *rest = offset
    dz = rest[0] if rest else 0.0
    numbers = int | float
    if isinstance(dx, numbers) and isinstance(dy, numbers) and isinstance(dz, numbers):
        return on_geometry(geometry, _translate(float(dx), float(dy), float(dz)))

    def column(d: float | IntoExprColumn) -> IntoExprColumn:
        return pl.lit(float(d)) if isinstance(d, int | float) else d

    has_dz = not isinstance(dz, int | float) or dz != 0.0
    by = point(column(dx), column(dy), z=column(dz) if has_dz else None)
    return on_geometry_pair(geometry, by, _translate_by)


Origin = tuple[float, float] | tuple[float, float, float]
_ZERO = (0.0, 0.0, 0.0)


def _degrees(amount: float, unit: Literal["deg", "pi"], name: str) -> float:
    """`amount` in degrees, refused while the expression is built if it is not a number."""
    if unit not in ("deg", "pi"):
        msg = f'`unit` has to be "deg" or "pi", got: {unit!r}'
        raise ValueError(msg)
    if not math.isfinite(amount):
        msg = f"`{name}` has to be a finite number, got: {amount!r}"
        raise ValueError(msg)
    return amount * 180.0 if unit == "pi" else amount


def _origin(origin: Origin) -> tuple[float, float, float]:
    """`origin` as `(x, y, z)`, a `z` left out being 0."""
    if len(origin) not in (2, 3) or not all(math.isfinite(at) for at in origin):
        msg = f"`origin` has to be two or three finite numbers, got: {origin!r}"
        raise ValueError(msg)
    x, y, *z = origin
    return x, y, z[0] if z else 0.0


def _about(
    linear: tuple[tuple[float, float, float], ...], origin: tuple[float, float, float]
) -> list[float]:
    """The affine matrix that applies the 3×3 `linear` part about `origin` instead of (0, 0, 0).

    That is moving `origin` to (0, 0, 0), applying it, and moving it back:
    L.(p - origin) + origin, which is L.p + constant offset origin - L.origin.
    """
    matrix: list[float] = []
    for row, at in zip(linear, origin, strict=True):
        moved = sum(r * o for r, o in zip(row, origin, strict=True))
        matrix += [*row, at - moved]
    return matrix


def _cos_sin(degrees: float) -> tuple[float, float]:
    """The cosine and sine of an angle, exact for whole quarter turns.

    `math.cos(math.pi / 2)` is 6e-17 rather than 0, which would leave a quarter turn
    of `(1, 0)` at `(6e-17, 1)` and make every row of the matrix read every coordinate.
    """
    quarters, rest = divmod(degrees, 90.0)
    if rest == 0.0:
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[int(quarters) % 4]
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def _tan(degrees: float, name: str) -> float:
    """The tangent of an angle, exact for whole eighth turns,
    refused where it does not exist (a quarter turn, either way).

    `math.tan(math.pi / 4)` is 0.9999999999999999 rather than 1.
    """
    eighths, rest = divmod(degrees, 45.0)
    if rest == 0.0:
        tangent = (0.0, 1.0, None, -1.0)[int(eighths) % 4]
        if tangent is None:
            msg = f"`{name}` cannot be a quarter turn (±90°): it would skew without end"
            raise ValueError(msg)
        return tangent
    return math.tan(math.radians(degrees))


def _affine_about(column: pl.Expr, origin: pl.Expr, matrix: Sequence[float]) -> pl.Expr:
    """`_affine`, but about each geometry's own point of `origin`,
    a `geoarrow.point` column, rather than about (0, 0, 0):
    M.(p - origin) + origin.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[column, origin],
        function_name="affine_about",
        is_elementwise=True,
        kwargs={"coefficients": [float(c) for c in matrix]},
    )


def _points(name: str, dtype: GeoArrowType) -> None:
    """Refuse a column given as `name` that does not hold points."""
    if not isinstance(dtype, PointType):
        msg = f"`{name}` has to be a `geoarrow.point` column, got: {dtype!r}"
        raise TypeError(msg)


def _about_centroid(
    name: str,
    build_about: Callable[[pl.Expr, GeoArrowType, pl.Expr, GeoArrowType], pl.Expr],
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    """An `on_geometry` callback that hands `build_about` (one of the `_*_about` builders)
    each geometry's own centroid as its origin.
    """

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # A multi-geometry has a list of centroids, one per part, not one to turn about.
        if not isinstance(geometry, LineStringType | PolygonType):
            msg = (
                f"cannot {name} about its centroid: expects a linestring "
                f"or polygon column, got: {geometry!r}"
            )
            raise TypeError(msg)
        return on_geometry_pair(column, _centroid(column, geometry), build_about)

    return build


def _rotation(
    degrees: float, axis: Literal["x", "y", "z"]
) -> tuple[tuple[float, float, float], ...]:
    cos, sin = _cos_sin(degrees)
    # Counter-clockwise for a positive angle, looking down the axis from its positive end.
    # fmt: off
    return {
        "x": (
            (1,    0,    0),
            (0,    cos, -sin),
            (0,    sin,  cos),
        ),
        "y": (
            (cos,  0,    sin),
            (0,    1,    0),
            (-sin, 0,    cos),
        ),
        "z": (
            (cos, -sin,  0),
            (sin,  cos,  0),
            (0,    0,    1),
        ),
    }[axis]
    # fmt: on


def _axis(axis: Literal["x", "y", "z"]) -> None:
    if axis not in ("x", "y", "z"):
        msg = f'`axis` has to be "x", "y" or "z", got: {axis!r}'
        raise ValueError(msg)


def _can_rotate(axis: Literal["x", "y", "z"], geometry: GeoArrowType) -> None:
    # Turning about `x` or `y` moves positions into and out of `z`,
    # which a geometry without one cannot hold.
    if axis != "z" and "z" not in geometry._dimension:
        msg = f"cannot rotate about the {axis} axis: {geometry!r} has no z coordinate"
        raise TypeError(msg)


def _rotate(
    degrees: float,
    axis: Literal["x", "y", "z"],
    origin: tuple[float, float, float],
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    matrix = _about(_rotation(degrees, axis), origin)

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        _can_rotate(axis, geometry)
        return _affine(column, matrix)

    return build


def _rotate_about(
    degrees: float, axis: Literal["x", "y", "z"]
) -> Callable[[pl.Expr, GeoArrowType, pl.Expr, GeoArrowType], pl.Expr]:
    rotation = _rotation(degrees, axis)

    def build(
        column: pl.Expr, geometry: GeoArrowType, origin: pl.Expr, points: GeoArrowType
    ) -> pl.Expr:
        _can_rotate(axis, geometry)
        _points("origin", points)
        return _affine_about(column, origin, _about(rotation, _ZERO))

    return build


@doc(verb="turn")
def rotate(
    geometry: IntoExprColumn,
    amount: float,
    unit: Literal["deg", "pi"] = "deg",
    axis: Literal["x", "y", "z"] = "z",
    origin: Origin | IntoExprColumn = (0.0, 0.0, 0.0),
) -> pl.Expr:
    """Turn every position of a geometry about an axis through `origin`.

    `amount` is in degrees, or with `unit="pi"` in multiples of π radians:
    `rotate("shape", 90)` and `rotate("shape", 0.5, unit="pi")` are the same quarter turn.
    A positive amount turns counter-clockwise, looking down the axis from its positive end
    towards the origin (the right-hand rule).

    By default this turns about the `z` axis: in the plane, leaving `z` as it is,
    so it works on any geometry.
    Turning about `x` or `y` needs a geometry with a `z`, and is refused for one without.
    `m`, being a measure, is always carried through untouched.

    `origin` is the point the axis runs through, by default `(0, 0, 0)`. It is either
    {origin_per_column_or_row}

    A `z` left out is 0.
    Coordinates are turned as they are, in the units of their CRS:
    in longitude/latitude, this does not turn anything on the globe,
    and can leave positions outside of the valid range.
    """
    _axis(axis)
    degrees = _degrees(amount, unit, "amount")
    # A column of points, rather than one point given as numbers.
    if isinstance(origin, str | pl.Expr | pl.Series):
        return on_geometry_pair(geometry, origin, _rotate_about(degrees, axis))
    return on_geometry(geometry, _rotate(degrees, axis, _origin(origin)))


@doc()
def rotate_about_centroid(
    geometry: IntoExprColumn,
    amount: float,
    unit: Literal["deg", "pi"] = "deg",
    axis: Literal["x", "y", "z"] = "z",
) -> pl.Expr:
    """Turn every geometry about an axis through its own centroid.

    This is `rotate(geometry, amount, unit, axis, origin=geo.centroid(geometry))`,
    see `rotate` for `amount`, `unit` and `axis`,
    and `centroid` for where the centroid of a linestring or polygon is.

    {about_centroid_accepts}


    ```python
    df.select(geo.rotate_about_centroid("parcel", 30))
    ```
    """

    _axis(axis)
    about = _rotate_about(_degrees(amount, unit, "amount"), axis)
    return on_geometry(geometry, _about_centroid("rotate", about))


def _shear(
    xs: float, ys: float, unit: Literal["deg", "pi"]
) -> tuple[tuple[float, float, float], ...]:
    tan_x = _tan(_degrees(xs, unit, "xs"), "xs")
    tan_y = _tan(_degrees(ys, unit, "ys"), "ys")
    # fmt: off
    return (
        (1,     tan_x, 0),
        (tan_y, 1,     0),
        (0,     0,     1),
    )
    # fmt: on


def _skew_about(
    shear: tuple[tuple[float, float, float], ...],
) -> Callable[[pl.Expr, GeoArrowType, pl.Expr, GeoArrowType], pl.Expr]:
    def build(
        column: pl.Expr, _: GeoArrowType, origin: pl.Expr, points: GeoArrowType
    ) -> pl.Expr:
        _points("origin", points)
        return _affine_about(column, origin, _about(shear, _ZERO))

    return build


@doc(verb="skew")
def skew(
    geometry: IntoExprColumn,
    xs: float = 0.0,
    ys: float = 0.0,
    unit: Literal["deg", "pi"] = "deg",
    origin: Origin | IntoExprColumn = (0.0, 0.0, 0.0),
) -> pl.Expr:
    """Shear every position of a geometry in the plane, by an angle along each axis.

    `xs` leans the geometry along `x`: every position moves along `x`
    by `tan(xs)` times how far it is from `origin` along `y`,
    so a line that ran parallel to the `y` axis ends up at an angle `xs` from it,
    leaning towards positive `x` for a positive angle.
    `ys` does the same along `y`, by `tan(ys)` times the distance along `x`.
    Both are applied to the positions as they were:

    ```text
    x' = x + tan(xs)·(y - origin_y)
    y' = y + tan(ys)·(x - origin_x)
    ```

    The angles are in degrees, or with `unit="pi"` in multiples of π radians.
    A quarter turn (±90°) has no tangent and is refused.

    This works on any geometry: `z` stays as it is,
    and `m`, being a measure, is always carried through untouched.

    `origin` is the point that stays where it is, by default `(0, 0, 0)`. It is either
    {origin_per_column_or_row}

    Coordinates are sheared as they are, in the units of their CRS.
    """
    shear = _shear(xs, ys, unit)
    # A column of points, rather than one point given as numbers.
    if isinstance(origin, str | pl.Expr | pl.Series):
        return on_geometry_pair(geometry, origin, _skew_about(shear))
    matrix = _about(shear, _origin(origin))
    return on_geometry(geometry, lambda column, _: _affine(column, matrix))


@doc()
def skew_about_centroid(
    geometry: IntoExprColumn,
    xs: float = 0.0,
    ys: float = 0.0,
    unit: Literal["deg", "pi"] = "deg",
) -> pl.Expr:
    """Shear every geometry in the plane about its own centroid.

    This is `skew(geometry, xs, ys, unit, origin=geo.centroid(geometry))`,
    see `skew` for `xs`, `ys` and `unit`,
    and `centroid` for where the centroid of a linestring or polygon is.

    {about_centroid_accepts}

    ```python
    df.select(geo.skew_about_centroid("parcel", xs=15))
    ```
    """
    about = _skew_about(_shear(xs, ys, unit))
    return on_geometry(geometry, _about_centroid("skew", about))


def _stretch(
    xfact: float, yfact: float, zfact: float
) -> tuple[tuple[float, float, float], ...]:
    # fmt: off
    return (
        (xfact, 0,     0),
        (0,     yfact, 0),
        (0,     0,     zfact),
    )
    # fmt: on


def _factors(xfact: float, yfact: float, zfact: float) -> None:
    for name, factor in (("xfact", xfact), ("yfact", yfact), ("zfact", zfact)):
        if not math.isfinite(factor):
            msg = f"`{name}` has to be a finite number, got: {factor!r}"
            raise ValueError(msg)


def _can_scale(zfact: float, geometry: GeoArrowType) -> None:
    # Without this, scaling an XY column by zfact would quietly do nothing.
    if zfact != 1.0 and "z" not in geometry._dimension:
        msg = f"cannot scale by zfact: {geometry!r} has no z coordinate"
        raise TypeError(msg)


def _scale(
    xfact: float, yfact: float, zfact: float, origin: tuple[float, float, float]
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    matrix = _about(_stretch(xfact, yfact, zfact), origin)

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        _can_scale(zfact, geometry)
        return _affine(column, matrix)

    return build


def _scale_about(
    xfact: float, yfact: float, zfact: float
) -> Callable[[pl.Expr, GeoArrowType, pl.Expr, GeoArrowType], pl.Expr]:
    stretch = _stretch(xfact, yfact, zfact)

    def build(
        column: pl.Expr, geometry: GeoArrowType, origin: pl.Expr, points: GeoArrowType
    ) -> pl.Expr:
        _can_scale(zfact, geometry)
        _points("origin", points)
        return _affine_about(column, origin, _about(stretch, _ZERO))

    return build


@doc(verb="scale")
def scale(
    geometry: IntoExprColumn,
    xfact: float = 1.0,
    yfact: float = 1.0,
    zfact: float = 1.0,
    origin: Origin | IntoExprColumn = (0.0, 0.0, 0.0),
) -> pl.Expr:
    """Stretch every position of a geometry away from (or towards) `origin`,
    by a factor along each axis:

    ```text
    x' = origin_x + xfact·(x - origin_x)
    ```

    and the same for `y` and `z`.
    A factor above 1 stretches, one between 0 and 1 shrinks,
    a negative one mirrors as well, and 0 flattens the geometry onto `origin` along that axis.

    A `zfact` other than 1 is refused for a geometry with no `z`,
    rather than silently ignored.
    `m`, being a measure, is always carried through untouched.

    `origin` is the point that stays where it is, by default `(0, 0, 0)`. It is either
    {origin_per_column_or_row}

    A `z` left out is 0.
    Coordinates are scaled as they are, in the units of their CRS.
    """
    _factors(xfact, yfact, zfact)
    # A column of points, rather than one point given as numbers.
    if isinstance(origin, str | pl.Expr | pl.Series):
        return on_geometry_pair(geometry, origin, _scale_about(xfact, yfact, zfact))
    return on_geometry(geometry, _scale(xfact, yfact, zfact, _origin(origin)))


@doc()
def scale_about_centroid(
    geometry: IntoExprColumn,
    xfact: float = 1.0,
    yfact: float = 1.0,
    zfact: float = 1.0,
) -> pl.Expr:
    """Stretch every geometry away from (or towards) its own centroid.

    This is `scale(geometry, xfact, yfact, zfact, origin=geo.centroid(geometry))`,
    see `scale` for the factors,
    and `centroid` for where the centroid of a linestring or polygon is.

    {about_centroid_accepts}

    ```python
    df.select(geo.scale_about_centroid("parcel", 2.0, 2.0))
    ```
    """
    _factors(xfact, yfact, zfact)
    about = _scale_about(xfact, yfact, zfact)
    return on_geometry(geometry, _about_centroid("scale", about))


def _can_transform(matrix: list[float], geometry: GeoArrowType) -> None:
    # Anything that reads `z` (c, f) or makes it (the bottom row) other than as it was.
    _, _, c, _, _, _, f, _, *z_row = matrix
    uses_z = c != 0.0 or f != 0.0 or z_row != [0.0, 0.0, 1.0, 0.0]
    # Without this, a matrix that works with `z` would quietly be half applied.
    if uses_z and "z" not in geometry._dimension:
        msg = f"cannot apply a matrix that uses z: {geometry!r} has no z coordinate"
        raise TypeError(msg)


def _affine_transform(
    matrix: list[float],
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        _can_transform(matrix, geometry)
        return _affine(column, matrix)

    return build


def _affine_transform_about(
    matrix: list[float],
) -> Callable[[pl.Expr, GeoArrowType, pl.Expr, GeoArrowType], pl.Expr]:
    def build(
        column: pl.Expr, geometry: GeoArrowType, origin: pl.Expr, points: GeoArrowType
    ) -> pl.Expr:
        _can_transform(matrix, geometry)
        _points("origin", points)
        return _affine_about(column, origin, matrix)

    return build


@doc(verb="transform")
def affine_transform(
    geometry: IntoExprColumn,
    matrix: Sequence[float],
    origin: Origin | IntoExprColumn = (0.0, 0.0, 0.0),
) -> pl.Expr:
    """Transform every position of a geometry by an affine matrix of your own.

    The matrix is given as a flat list in the same order as shapely and GeoPandas take it,
    so a matrix written for them works here unchanged.
    Six numbers transform in the plane, leaving `z` as it is:

    ```text
    [a, b, d, e, xoff, yoff]

    x' = a·x + b·y + xoff
    y' = d·x + e·y + yoff
    ```

    Twelve numbers transform in space:

    ```text
    [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]

    x' = a·x + b·y + c·z + xoff
    y' = d·x + e·y + f·z + yoff
    z' = g·x + h·y + i·z + zoff
    ```

    A twelve-number matrix that reads `z` (`c` or `f` not 0)
    or changes it (a bottom row other than `0, 0, 1, 0`)
    is refused for a geometry with no `z`, rather than silently half applied.
    `m`, being a measure, is always carried through untouched.

    `origin` is the point the matrix is applied about, by default `(0, 0, 0)`:
    positions are taken relative to it, transformed, and put back,
    `x' = a·(x - origin_x) + b·(y - origin_y) + c·(z - origin_z) + xoff + origin_x`
    and likewise for `y` and `z`. The default leaves the matrix as shapely applies it.
    It is either
    {origin_per_column_or_row}

    A `z` left out is 0.
    `translate`, `rotate`, `skew` and `scale` are each one of these,
    with the matrix worked out for you.
    Coordinates are transformed as they are, in the units of their CRS.
    """
    numbers = [float(n) for n in matrix]
    if len(numbers) not in (6, 12) or not all(math.isfinite(n) for n in numbers):
        msg = f"`matrix` has to be 6 or 12 finite numbers, got: {matrix!r}"
        raise ValueError(msg)
    if len(numbers) == 6:
        a, b, d, e, xoff, yoff = numbers
        c = f = g = h = zoff = 0.0
        i = 1.0
    else:
        a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = numbers
    # fmt: off
    rows = [
        a, b, c, xoff,
        d, e, f, yoff,
        g, h, i, zoff,
    ]
    # fmt: on
    # A column of points, rather than one point given as numbers.
    if isinstance(origin, str | pl.Expr | pl.Series):
        return on_geometry_pair(geometry, origin, _affine_transform_about(rows))

    linear = ((a, b, c), (d, e, f), (g, h, i))
    about = _about(linear, _origin(origin))
    # Its own offsets on top of those that move it about `origin`.
    for k, offset in enumerate((xoff, yoff, zoff)):
        about[4 * k + 3] += offset
    return on_geometry(geometry, _affine_transform(about))
