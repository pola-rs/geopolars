# geopolars-core

Picks the kernel for each kind of geometry.
It wraps a chunk of a geometry column's storage in the [`geopolars-views`](../geopolars-views/README.md) view of its `Kind`,
and calls the [`geopolars-compute`](../geopolars-compute/README.md) kernel for that view.
