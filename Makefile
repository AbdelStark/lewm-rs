PYTHON ?= python3
CARGO_AUDIT_DB ?= target/advisory-db/cargo-audit

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
	cargo bench --workspace --benches

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

check: fmt lint
	cargo check --workspace --all-targets
	$(PYTHON) scripts/check_layers.py
	$(PYTHON) scripts/check_specs.py
	$(PYTHON) scripts/check_jobs.py
	$(PYTHON) scripts/check_otel_infra.py
	$(PYTHON) scripts/check_train_so100_job.py
	$(PYTHON) scripts/check_nondet.py
	$(PYTHON) -m py_compile python/hf_pricing.py python/cost_ledger.py python/upload_checkpoints.py scripts/launch_hf_job.py
	$(PYTHON) python/cost_ledger.py check --path reports/cost.md --cap-usd 200
	cargo deny check
	# hdf5-metno depends on paste; cargo-deny still blocks direct workspace unmaintained deps.
	# Tract 0.22.1 pins the vulnerable time dependency; keep the audit ignore scoped until tract can upgrade.
	# Burn 0.20.1 pulls bincode 2.0.1; ADR 0002 tracks the date-bounded waiver.
	cargo audit --db "$(CARGO_AUDIT_DB)" --deny warnings --ignore RUSTSEC-2024-0436 --ignore RUSTSEC-2026-0009 --ignore RUSTSEC-2025-0141

accept: check test docs
	@if [ -d python ] && [ -f python/Makefile ]; then \
		$(MAKE) -C python check; \
	else \
		printf '%s\n' 'python check skipped: python/Makefile not present'; \
	fi
	$(PYTHON) scripts/check_hub_artifacts.py
	@if [ -x scripts/check_release_inventory.sh ]; then \
		scripts/check_release_inventory.sh; \
	else \
		printf '%s\n' 'release inventory skipped: scripts/check_release_inventory.sh not present'; \
	fi

clean:
	cargo clean
