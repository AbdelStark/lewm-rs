PYTHON ?= python3
RUFF ?= ruff
CARGO_AUDIT_DB ?= target/advisory-db/cargo-audit

.PHONY: fmt lint test test-fast bench docs docsite check accept clean py-lint

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

# Build the mdBook documentation site under docs/.
# Requires mdbook on PATH (`cargo install mdbook`).
docsite:
	@if command -v mdbook >/dev/null 2>&1; then \
		printf '%s\n' "mdbook build docs/"; \
		mdbook build docs; \
		mdbook test docs; \
	else \
		printf '%s\n' 'mdbook not installed; install with: cargo install mdbook'; \
		exit 1; \
	fi

# Lint Python helpers with Ruff when available. Ruff is wired through
# `python/pyproject.toml`; falls back to a `py_compile` syntax sweep when the
# binary is missing so the gate degrades gracefully on minimal environments.
py-lint:
	@if command -v $(RUFF) >/dev/null 2>&1; then \
		echo "ruff check ..."; \
		$(RUFF) check --config python/pyproject.toml python scripts; \
	else \
		printf '%s\n' 'ruff not installed; falling back to py_compile only (install with: pip install ruff)'; \
	fi

check: fmt lint py-lint
	cargo check --workspace --all-targets
	$(PYTHON) scripts/check_layers.py
	$(PYTHON) scripts/check_specs.py
	$(PYTHON) scripts/check_jobs.py
	$(PYTHON) scripts/check_otel_infra.py
	$(PYTHON) scripts/check_train_so100_job.py
	$(PYTHON) scripts/check_nondet.py
	$(PYTHON) scripts/check_release_blockers.py --allow-open
	$(PYTHON) scripts/check_phase_a_handoff.py
	$(PYTHON) scripts/check_phase_a_approval.py
	$(PYTHON) scripts/check_f1_source_build_dry_run_report.py
	$(PYTHON) scripts/check_runtime_image_publish_report.py
	$(PYTHON) scripts/check_full_pusht_contract_smoke_report.py
	$(PYTHON) scripts/check_pusht_full_safetensors_hub_audit_report.py
	$(PYTHON) scripts/check_pusht_warmstart_source_smoke_report.py
	$(PYTHON) scripts/check_pusht_warmstart_hub_audit_report.py
	$(PYTHON) -m py_compile python/hf_pricing.py python/cost_ledger.py python/upload_checkpoints.py python/param_name_map.py python/convert_reference.py python/verify_conversion.py scripts/launch_hf_job.py scripts/check_release_blockers.py scripts/check_phase_a_handoff.py scripts/check_phase_a_approval.py scripts/upload_model_cards.py scripts/check_warmstart_source.py scripts/full_pusht_contract_smoke.py scripts/check_full_pusht_contract_smoke_report.py scripts/check_f1_source_build_dry_run_report.py scripts/check_runtime_image_publish_report.py scripts/f1_export_pusht_onnx.py scripts/audit_pusht_full_safetensors.py scripts/check_pusht_full_safetensors_hub_audit_report.py scripts/pusht_warmstart_source_smoke.py scripts/check_pusht_warmstart_source_smoke_report.py scripts/audit_pusht_warmstart_sources.py scripts/check_pusht_warmstart_hub_audit_report.py
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
	$(PYTHON) scripts/check_release_blockers.py
	$(PYTHON) scripts/check_hub_artifacts.py
	@if [ -x scripts/check_release_inventory.sh ]; then \
		scripts/check_release_inventory.sh; \
	else \
		printf '%s\n' 'release inventory skipped: scripts/check_release_inventory.sh not present'; \
	fi

clean:
	cargo clean
