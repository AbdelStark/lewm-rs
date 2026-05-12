PYTHON ?= python3

.PHONY: fmt lint test test-fast bench docs check accept clean

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace --all-features

test-fast:
	cargo test --workspace --lib --bins -- --skip "_slow_"

bench:
	cargo bench --workspace

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

check: fmt lint
	cargo check --workspace --all-targets
	$(PYTHON) scripts/check_layers.py
	$(PYTHON) scripts/check_specs.py
	cargo deny check
	cargo audit --deny warnings

accept: check test docs
	@if [ -d python ] && [ -f python/Makefile ]; then \
		$(MAKE) -C python check; \
	else \
		printf '%s\n' 'python check skipped: python/Makefile not present'; \
	fi
	@if [ -x scripts/check_release_inventory.sh ]; then \
		scripts/check_release_inventory.sh; \
	else \
		printf '%s\n' 'release inventory skipped: scripts/check_release_inventory.sh not present'; \
	fi

clean:
	cargo clean
