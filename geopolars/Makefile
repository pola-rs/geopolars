.DEFAULT_GOAL := help

SHELL=/bin/bash
ifeq ($(VENV),)
VENV := ../.venv
endif
VENV_BIN=$(VENV)/bin

.PHONY: build
# dev build
build:
	@$(MAKE) -s -C .. install

.PHONY: build-release
build-release:
	@$(MAKE) -s -C .. install-release

.PHONY: test
## Run fast unittests
test: build
	$(VENV_BIN)/pytest -n auto $(PYTEST_ARGS)

.PHONY: test-all
## Run all tests, including the slow ones
test-all: build
	$(VENV_BIN)/pytest -n auto -m "slow or not slow" $(PYTEST_ARGS)

.PHONY: clean
clean:
	@rm -rf .pytest_cache/
	@find . -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -exec rm -r {} +
