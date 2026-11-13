SHELL=/bin/bash

.PHONY: install install-release wheel polars-src run-geo run-line run-mean-coordinate run-mean-coordinate-manual run-crs run-parquet rebuild test test-rust test-python clean

# The polars commit we build against. Cargo.toml is the source of truth;
# this checks out the same commit for uv to build Python polars from.
POLARS_REV := $(shell sed -n 's/^polars = .*rev = "\([0-9a-f]*\)".*/\1/p' Cargo.toml)
POLARS_DIR := .polars
POLARS_STAMP := $(POLARS_DIR)/.rev-$(POLARS_REV)

# uv only notices a path source changed if its pyproject.toml did,
# so after a new checkout force it to rebuild polars.
UV_SYNC = if [ -e $(POLARS_DIR)/.reinstall ]; then \
		reinstall="--reinstall-package polars --reinstall-package polars-runtime-32"; \
	fi; \
	uv sync $$reinstall $(UV_SYNC_ARGS) && rm -f $(POLARS_DIR)/.reinstall

# Sync the venv, building the plugin unoptimized (fast edit-compile loop).
# this builds polars as debug.
#
# uv can take the plugin to be up to date after Rust edits, and keep a stale build,
# so always rebuild it.
# In uv's isolated build env, maturin points PYO3_PYTHON at a fresh temporary interpreter,
# so cargo recompiles pyo3 and all of polars on top of it every time.
# Building in our own venv keeps that path, and so cargo's cache, stable.
# A fresh venv has no maturin yet to build with, so that first build stays isolated.
install: UV_SYNC_ARGS = --reinstall-package geopolars \
	$(if $(wildcard .venv/bin/maturin),--no-build-isolation-package geopolars)
install: polars-src
	export MATURIN_PEP517_ARGS="--profile dev"; $(UV_SYNC)

# Sync the venv, building the plugin optimized.
install-release: polars-src
	$(UV_SYNC)

WHEEL_SRC := target/wheel-src

## Build a release wheel into target/wheels, with the `@doc` docstrings filled in,
## so that editors show them.
## maturin cannot rewrite Python files itself, so this builds from a copy of the tracked files.
wheel:
	rm -rf $(WHEEL_SRC) && mkdir -p $(WHEEL_SRC)
	git ls-files -z | tar --null -T - -cf - | tar -xf - -C $(WHEEL_SRC)
	python3 scripts/expand_docstrings.py $(WHEEL_SRC)/geopolars/geopolars
	cd $(WHEEL_SRC)/geopolars && CARGO_TARGET_DIR=$(CURDIR)/target \
		$(CURDIR)/.venv/bin/maturin build --release --out $(CURDIR)/target/wheels

## Check out POLARS_REV into POLARS_DIR, if it isn't already.
polars-src: $(POLARS_STAMP)

# Fetches only the one commit, not the whole polars history.
$(POLARS_STAMP):
	@mkdir -p $(POLARS_DIR)
	@[ -d $(POLARS_DIR)/.git ] || git -C $(POLARS_DIR) init -q
	git -C $(POLARS_DIR) fetch --depth 1 https://github.com/pola-rs/polars.git $(POLARS_REV)
	git -C $(POLARS_DIR) checkout -q --force FETCH_HEAD
	@rm -f $(POLARS_DIR)/.rev-*
	@touch $(POLARS_DIR)/.reinstall $@

## Force a rebuild of just the plugin, ignoring uv's cache.
rebuild: polars-src
	uv sync --reinstall-package geopolars

# Rust tests can be a bit slower, so only run them if we ask specifically.
test: test-python

test-all: test-rust test-python

## Rust unit tests. 
## Run with --no-default-features  so pyo3's `extension-module` is dropped
test-rust:
	cargo test -p geopolars -p geopolars-views -p geopolars-compute -p geopolars-core --no-default-features

## Python tests. Delegates to the package Makefile, which builds first.
test-python:
	@$(MAKE) -s -C geopolars test

## Run the geoarrow.point proof of concept.
run-geo: install
	uv run examples/run_geo.py

## Run the geoarrow.linestring proof of concept.
run-line: install
	uv run examples/run_line.py

## Run the CRS reprojection example, for points and lines.
run-crs: install
	uv run examples/run_crs.py

## Run the GeoParquet reading and writing example.
run-parquet: install
	uv run examples/run_parquet.py

## Run the mean coordinate proof of concept.
run-mean-coordinate: install
	uv run examples/run_mean_coordinate.py

## Run the same mean coordinate written by hand, in plain Polars.
run-mean-coordinate-manual: install
	uv run examples/run_mean_coordinate_manual.py

clean:
	-@rm -rf .venv target
	-@rm -f uv.lock
