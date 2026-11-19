# Contributing

## Setting up

Build GeoPolars from source, as described under *Compile GeoPolars from source* in the
[README](../../../../README.md#installation).
`make install` sets up a virtual environment in `.venv`,
and builds GeoPolars and the Polars commit it is pinned to into it.

Run `make help` for every target.
The ones you'll find yourself needing most:

- `make test`: the Python tests, without the slow ones.
- `make test-all`: the Rust and Python tests, including the slow ones.
- `make docs`: the API reference, into `target/docs/html`.

## Developer documentation

These describe the design and the architecture of GeoPolars.

- [A high-level overview of the architecture](../architecture.md)
- [What happens when you call an expression, at plan and execution time](../typical-expression-flow.md),
  accompanied by [a diagram](../expression-flow.dot)
- [What to consider when implementing new or editing existing expressions](../expression-implementation-guide.md)
- [The views kernels take, in `geopolars-views`](../../../../geopolars-views/README.md)
- [What to consider when writing kernels, in `geopolars-compute`](../../../../geopolars-compute/README.md),
  with separate guides for [native kernels](../native-kernel-implementation-guide.md)
  and [kernels forwarding to GEOS](../geos-kernel-implementation-guide.md)
  on top of the [general kernel guide](../kernel-implementation-guide.md)
- [Which expression is implemented at which *tier*](../tier-per-expression.md)
