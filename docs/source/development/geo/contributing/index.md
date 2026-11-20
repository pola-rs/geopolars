# Contributing

## Setting up

Build GeoPolars from source, as described under *Compile GeoPolars from source* in the
[README](https://github.com/pola-rs/geopolars/blob/main/README.md#installation).

Run `make help` for every target.
The ones you'll find yourself needing most:

- `make test`: the Python tests, without the slow ones.
- `make test-all`: the Rust and Python tests, including the slow ones.
- `make docs-api`: the API reference, into `target/docs/html`.
- `make docs-guide`: the user and developer guides, into `target/docs-guide/site`.
- `make docs-guide-serve`: the guides on http://localhost:8000, rebuilt as you edit them.
- `make docs`: both.

## Developer documentation

These describe the design and the architecture of GeoPolars.

- [A high-level overview of the architecture](../architecture.md)
- [What happens when you call an expression, at plan and execution time](../typical-expression-flow.md),
  accompanied by [a diagram](https://github.com/pola-rs/geopolars/blob/main/docs/source/development/geo/expression-flow.dot)
- [What to consider when implementing new or editing existing expressions](../expression-implementation-guide.md)
- [The views kernels take, in `geopolars-views`](https://github.com/pola-rs/geopolars/blob/main/geopolars-views/README.md)
- [What to consider when writing kernels, in `geopolars-compute`](https://github.com/pola-rs/geopolars/blob/main/geopolars-compute/README.md),
  with separate guides for [native kernels](../native-kernel-implementation-guide.md)
  and [kernels forwarding to GEOS](../geos-kernel-implementation-guide.md)
  on top of the [general kernel guide](../kernel-implementation-guide.md)
- [Which expression is implemented at which *tier*](../tier-per-expression.md)
