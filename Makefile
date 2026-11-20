SHELL=/bin/bash

.PHONY: help install install-release wheel docs docs-html docs-guide docs-guide-build polars-src test test-all test-rust test-python test-python-all clean

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

## Sync the venv, building the plugin unoptimized (fast edit-compile loop).
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

## Sync the venv, building the plugin optimized.
install-release: polars-src
	$(UV_SYNC)

WHEEL_SRC := target/wheel-src
DOCS_PKG := target/docs-pkg

## Build a release wheel into target/wheels, with the `@doc` docstrings filled in,
## so that editors show them.
# maturin cannot rewrite Python files itself, so this builds from a copy of the tracked files.
wheel: | .venv/bin/maturin
	rm -rf $(WHEEL_SRC) && mkdir -p $(WHEEL_SRC)
	git ls-files -z | tar --null -T - -cf - | tar -xf - -C $(WHEEL_SRC)
	.venv/bin/python scripts/expand_docstrings.py $(WHEEL_SRC)/geopolars/geopolars
	cd $(WHEEL_SRC)/geopolars && CARGO_TARGET_DIR=$(CURDIR)/target \
		$(CURDIR)/.venv/bin/maturin build --release --out $(CURDIR)/target/wheels

## Build the HTML API reference into target/docs/html.
# The docs tooling is not a default group, so `install` leaves it out
docs: install
	uv sync --inexact --only-group docs
	$(MAKE) docs-html

## Build the HTML API reference into target/docs/html, with the venv as it is.
# Sphinx imports a copy of the package with the `@doc` docstrings filled in, as in the wheel,
# which goes before the editable install on the path.
docs-html:
	rm -rf $(DOCS_PKG) && mkdir -p $(DOCS_PKG)
	cp -r geopolars/geopolars $(DOCS_PKG)/
	.venv/bin/python scripts/expand_docstrings.py $(DOCS_PKG)/geopolars
	PYTHONPATH=$(DOCS_PKG) .venv/bin/sphinx-build -M html geopolars/docs/source target/docs $(SPHINXOPTS)

GUIDE_DIR := target/docs-guide

## Build the user and developer guides with zensical into target/docs-guide/site.
## It fails on broken links, unknown API keys, and failing examples, as the Polars docs build would.
docs-guide: install
	grep -E '^(zensical|markdown-exec)==' $(POLARS_DIR)/docs/source/requirements.txt \
		| uv pip install tomli-w -r -
	$(MAKE) docs-guide-build

## Build the user and developer guides with zensical, with the venv as it is.
# zensical only logs a failing example, or an unknown API key in a `code_block`, so the build log is checked for them.
docs-guide-build: polars-src
	.venv/bin/python scripts/stage_docs_guide.py $(POLARS_DIR) $(GUIDE_DIR)
	set -o pipefail; cd $(GUIDE_DIR) && $(CURDIR)/.venv/bin/zensical build 2>&1 | tee build.log
	@if grep -E 'Execution of python code block exited with errors|Could not find' $(GUIDE_DIR)/build.log; then \
		echo "A code block failed, or names an unknown API key, see the build log above." >&2; exit 1; \
	fi

## Check out POLARS_REV into POLARS_DIR, if it isn't already.
polars-src: $(POLARS_STAMP)

# The wheel needs maturin from the venv, but not an up-to-date plugin.
.venv/bin/maturin: | polars-src
	$(UV_SYNC)

# Fetches only the one commit, not the whole polars history.
$(POLARS_STAMP):
	@mkdir -p $(POLARS_DIR)
	@[ -d $(POLARS_DIR)/.git ] || git -C $(POLARS_DIR) init -q
	git -C $(POLARS_DIR) fetch --depth 1 https://github.com/pola-rs/polars.git $(POLARS_REV)
	git -C $(POLARS_DIR) checkout -q --force FETCH_HEAD
	@rm -f $(POLARS_DIR)/.rev-*
	@touch $(POLARS_DIR)/.reinstall $@

## Python tests, without the slow ones.
# Rust tests can be a bit slower, so only run them if we ask specifically.
test: test-python

## Rust and Python tests, including the slow ones.
test-all: test-rust test-python-all

## Rust unit tests.
# Run with --no-default-features so pyo3's `extension-module` is dropped
test-rust:
	cargo test -p geopolars -p geopolars-views -p geopolars-compute -p geopolars-core --no-default-features

# pytest runs from the package, so that it picks up the config in its pyproject.toml.
PYTEST := cd geopolars && $(CURDIR)/.venv/bin/pytest -n auto

## Python tests, without the slow ones.
test-python: install
	$(PYTEST) $(PYTEST_ARGS)

## All Python tests, including the slow ones.
test-python-all: install
	$(PYTEST) -m "slow or not slow" $(PYTEST_ARGS)

## Remove the venv and all build output.
clean:
	-@rm -rf .venv target geopolars/.pytest_cache
	-@find geopolars -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -exec rm -r {} +

## Show the targets and what they do.
help:
	@awk '/^## /{doc = doc (doc ? "\n" sprintf("%28s", "") : "") substr($$0, 4); next} /^#/{next} \
		/^[a-z-]+:/ && doc {printf "%-28s%s\n", substr($$0, 1, index($$0, ":") - 1), doc} {doc = ""}' $(MAKEFILE_LIST)
