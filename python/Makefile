# Python edge-helper gate for lewm-rs.
#
# The top-level `Makefile`'s `accept` target invokes `make -C python check` when
# this file is present (see RFC 0011 §6). Keeping the Python gate scoped to the
# `python/` subtree lets the Rust workspace remain the authoritative build, with
# Python serving only as an edge surface for parity, export, decoding, and Hub
# upload helpers.
#
# Conventions:
#   - Ruff is configured by `pyproject.toml` (see `[tool.ruff]`).
#   - All targets honour `PYTHON` (defaults to `python3`) so CI can pin a
#     specific interpreter without editing this file.
#   - `make check` is non-destructive; `make fix` writes Ruff auto-fixes in place.

PYTHON ?= python3
RUFF ?= ruff
PACKAGE_DIRS := . tests

.PHONY: check lint fix test compile clean

# Default target: run the full Python gate.
check: lint compile
	@echo "python check: OK"

# Lint via Ruff. Fails on any diagnostic; formatting is intentionally not
# enforced (see comment in `pyproject.toml`).
lint:
	$(RUFF) check $(PACKAGE_DIRS)

# Apply Ruff's auto-fixable suggestions in place. Use locally; CI should run
# `make check` instead.
fix:
	$(RUFF) check --fix $(PACKAGE_DIRS)

# Byte-compile every helper to catch syntax regressions cheaply.
# Mirrors the `py_compile` step the root Makefile already runs.
compile:
	$(PYTHON) -m py_compile $$(find . -maxdepth 2 -name "*.py" -not -path "./.venv/*")

# Pytest is opt-in: it requires the `parity` and `sim` optional dependencies
# to import test fixtures cleanly. Run with `uv run --extra parity pytest`.
test:
	$(PYTHON) -m pytest tests

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ tests/__pycache__
