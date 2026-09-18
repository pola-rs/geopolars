SHELL=/bin/bash

.PHONY: install install-release run run-geo run-line run-centroid run-release rebuild test test-rust test-python clean

## Sync the venv, building the plugin unoptimized (fast edit-compile loop).
install:
	MATURIN_PEP517_ARGS="--profile dev" uv sync

## Sync the venv, building the plugin optimized. 
install-release:
	uv sync

## Force a rebuild of just the plugin, ignoring uv's cache.
rebuild:
	uv sync --reinstall-package geopolars

test: test-rust test-python

## Rust unit tests. 
## Run with --no-default-features  so pyo3's `extension-module` is dropped
test-rust:
	cargo test -p geopolars --no-default-features

## Python tests. Delegates to the package Makefile, which builds first.
test-python:
	@$(MAKE) -s -C geopolars test

run: install
	uv run examples/run.py

## Run the geoarrow.point proof of concept.
run-geo: install
	uv run examples/run_geo.py

## Run the geoarrow.linestring proof of concept.
run-line: install
	uv run examples/run_line.py

## Run the coordinate centroid proof of concept.
run-centroid: install
	uv run examples/run_centroid.py

clean:
	-@rm -rf .venv target
	-@rm -f uv.lock
