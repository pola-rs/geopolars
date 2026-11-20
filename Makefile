SHELL=/bin/bash

.PHONY: help install install-release wheel docs docs-api docs-api-build docs-guide docs-guide-build docs-guide-serve docs-guide-stage polars-src test test-all test-rust test-python test-python-all clean

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

## Build the API reference and the guides.
docs: docs-api docs-guide

## Build the HTML API reference with Sphinx into target/docs/html.
# The docs tooling is not a default group, so `install` leaves it out
docs-api: install
	uv sync --inexact --only-group docs
	$(MAKE) docs-api-build

## Build the HTML API reference with Sphinx, with the venv as it is.
# Sphinx imports a copy of the package with the `@doc` docstrings filled in, as in the wheel,
# which goes before the editable install on the path.
docs-api-build:
	rm -rf $(DOCS_PKG) && mkdir -p $(DOCS_PKG)
	cp -r geopolars/geopolars $(DOCS_PKG)/
	.venv/bin/python scripts/expand_docstrings.py $(DOCS_PKG)/geopolars
	PYTHONPATH=$(DOCS_PKG) .venv/bin/sphinx-build -M html geopolars/docs/source target/docs $(SPHINXOPTS)

GUIDE_DIR := target/docs-guide
ZENSICAL := $(CURDIR)/.venv/bin/zensical
API_LINKS := docs/source/_build/API_REFERENCE_LINKS.yml
INSTALL_GUIDE_DEPS = grep -E '^(zensical|markdown-exec)==' $(POLARS_DIR)/docs/source/requirements.txt \
	| uv pip install --python .venv/bin/python -r -
# zensical only logs a failing example, an unknown API key, or any other warning.
GUIDE_PROBLEMS := ^(Warning|Error)\b|^[0-9]+ issues? found|code block exited with|Could not find

## Build the user and developer guides with zensical into target/docs-guide/site.
## It fails on broken links and failing examples, as the Polars docs CI does,
## and also on unknown API keys and macros that raise, which Polars lets through.
docs-guide: install
	$(INSTALL_GUIDE_DEPS)
	$(MAKE) docs-guide-build

## Serve the guides with zensical on localhost:8000.
## It serves a copy of the pages, so restart it to see your edits.
docs-guide-serve: install
	$(INSTALL_GUIDE_DEPS)
	$(MAKE) docs-guide-stage
	cd $(GUIDE_DIR) && $(ZENSICAL) serve

## Build the user and developer guides with zensical, with the venv as it is.
docs-guide-build: docs-guide-stage
	set -o pipefail; cd $(GUIDE_DIR) && $(ZENSICAL) build 2>&1 | tee build.log
	@if grep -E '$(GUIDE_PROBLEMS)' $(GUIDE_DIR)/build.log; then echo "The guides have the problems above." >&2; exit 1; fi

# Lays out the guides as in the Polars repo, with the zensical config and `_build` from Polars.
# Only our pages are there, as building the Polars pages would run all of their examples.
docs-guide-stage: polars-src
	rm -rf $(GUIDE_DIR) && mkdir -p $(GUIDE_DIR)/docs/{source,assets/data}
	cp -r $(POLARS_DIR)/docs/source/_build docs/source/{user-guide,development,src} $(GUIDE_DIR)/docs/source/
# Our API links go into the Polars `API_REFERENCE_LINKS.yml`, under `python:`.
	grep '^  ' $(API_LINKS) | sed -i '/^python:/r /dev/stdin' $(GUIDE_DIR)/$(API_LINKS)
# Our nav replaces the Polars nav.
# The Polars redirects go, as some of them lead to Polars pages we don't copy.
# Without `on_error_fail`, a page whose macro raises keeps its `{{code_block(...)}}` as text.
	sed -e '/^nav = \[/r docs/nav.toml' -e '/^nav = \[/,/^\]/d' \
		-e '/^\[project.plugins.redirects/,$$d' \
		-e '/^\[project.plugins.macros\]/a on_error_fail = true' \
		$(POLARS_DIR)/zensical.toml > $(GUIDE_DIR)/zensical.toml

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
