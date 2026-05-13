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
	$(PYTHON) scripts/check_jobs.py
	$(PYTHON) scripts/check_train_so100_job.py
	$(PYTHON) scripts/check_nondet.py
	$(PYTHON) -m py_compile python/hf_pricing.py python/cost_ledger.py
	$(PYTHON) python/cost_ledger.py check --path reports/cost.md --cap-usd 200
	cargo deny check
	# hdf5-metno depends on paste; cargo-deny still blocks direct workspace unmaintained deps.
	cargo audit --deny warnings --ignore RUSTSEC-2024-0436

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
