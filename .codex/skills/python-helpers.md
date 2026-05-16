---
name: python-helpers
description: Conventions for the `python/` edge helpers and `scripts/` validators. Activate when editing any `.py` file, when `make py-lint` fails, when adding new helpers, or when running `make -C python check`. The Python edge is Ruff-linted with a focused rule family and uses `uv`-managed dependencies.
prerequisites: Python 3.13 on PATH. Optional: `ruff` (graceful degradation to `py_compile` when missing). Optional: `uv` for full dep resolution.
---

# Python Helpers

<purpose>
Python in this repo is the edge for: ONNX export, reference-checkpoint conversion, dataset decoding, model-card upload, cost accounting, and benchmark plotting. It is NOT the training stack. This skill keeps the helpers idiomatic, lint-clean, and CI-safe.
</purpose>

<context>
- **Layout**:
  - `python/` — helpers (`export_onnx.py`, `convert_reference.py`, `verify_conversion.py`, `param_name_map.py`, `decode_so100_to_h5.py`, `compute_stats.py`, `compute_so100_stats.py`, `eval_compare.py`, `cost_ledger.py`, `hf_pricing.py`, `upload_checkpoints.py`, `upload_model_cards.py`, `plot_curves.py`, `pusht_runner.py`, `build_parity_fixture.py`, `model_cards/`, `tests/`).
  - `scripts/` — validators and operations (`check_specs.py`, `check_layers.py`, `check_jobs.py`, `check_nondet.py`, `check_otel_infra.py`, `check_train_so100_job.py`, `check_secrets.py`, `check_hub_artifacts.py`, `launch_hf_job.py`, `otel_smoke.py`, `bench_to_report.py`, `sbom.py`, `upload_model_cards.py`, `run_local.sh`, `build_reproducible_release.sh`, `verify_reproducible_release.sh`).
- **Lint**: Ruff via `python/pyproject.toml`. Rule families: `E`, `F`, `W`, `B`, `UP`, `SIM`, `RUF`, `I`.
- **Gate**: `make py-lint` runs Ruff from the repo root. `make -C python check` runs Ruff inside `python/` (used by `make accept`).
- **Python version**: `requires-python = ">=3.13,<4"`. Use modern syntax (`str | None`, `match`, structural pattern matching, etc.).
- **Deps**: declared in `python/pyproject.toml` (`av`, `blake3`, `h5py`, `numpy`, `pillow`, `pyarrow`; optional groups `sim`, `parity`). `dev` group: `pytest`, `ruff`.
- **Index**: PyTorch wheels come from `https://download.pytorch.org/whl/cpu` via `uv` index config.
- **Subprocess style**: helpers prefer `subprocess.run([…], check=True)` lists; avoid `shell=True`.
- **HF interaction**: prefer `huggingface_hub` library (Python) when in a script that already pulls `parity` extras; otherwise call the `hf` CLI from `scripts/launch_hf_job.py` / shell helpers.
</context>

<procedure>
**Editing existing helpers:**

1. Read the file fully — many helpers have implicit contracts (e.g., `export_onnx.py` infers `action_dim` from a Conv1d weight shape, not from the raw arg).
2. Run Ruff locally first:
   ```
   ruff check --config python/pyproject.toml python scripts
   ```
3. Fix diagnostics — `ruff check --fix` for autofixable; do not silence with `# noqa` without a specific rule code AND comment.
4. Re-run `make py-lint`, then `make check`.

**Adding a new helper:**

1. Decide location: validators / launchers / VCS hygiene → `scripts/`; ML / data / cost / artifacts → `python/`.
2. File header:
   ```python
   #!/usr/bin/env python3
   """One-line summary. Longer rationale here if useful."""
   from __future__ import annotations
   ```
3. CLI helpers use `argparse` (no `click` / `typer`).
4. Imports: stdlib → third-party → first-party (Ruff `I` enforces; `param_name_map` and siblings are declared first-party so import order is stable from either CWD).
5. Errors: raise `SystemExit` with a non-zero code on failure; reserve `RuntimeError` / domain-specific exceptions for libraries.
6. New deps: add to `python/pyproject.toml`, then `uv lock` (or document why a manual lockfile edit was needed) — and confirm `make accept`'s `make -C python check` still passes.
7. Tests: drop into `python/tests/` if the helper has non-trivial logic.

**Subprocess + shell**:

- Use `subprocess.run(["cmd", "arg"], check=True, text=True, capture_output=True)`.
- Never `shell=True` with user input.
- Use `pathlib.Path` for paths; avoid `os.path.join`.

**Cost ledger flow** (touching `cost_ledger.py` or `reports/cost.md`):

- See `hf-jobs-cost.md`. The ledger format is rigid; the validator parses fixed columns.
</procedure>

<patterns>
<do>
— `zip(a, b, strict=True)` — prevents silent length-mismatch (the export-onnx fix in #218).
— `raise SystemExit(130) from None` on `KeyboardInterrupt` — clean shutdown semantics.
— `contextlib.suppress(SomeError):` over bare `try/except/pass`.
— `[*a, b]` list-construction over `a + [b]` (Ruff prefers this).
— Type hints (`def fn(x: int) -> str:`) on every public function.
— `if __name__ == "__main__": main()` entry point; main returns `int` exit code.
</do>
<dont>
— Don't import `lewm_*` Python modules from `scripts/` — `scripts/` is intentionally self-contained.
— Don't use `print()` for diagnostic logs from a library function; use `logging` or return a structured result.
— Don't depend on torch/transformers in `python/` core helpers — they're in the `parity` extra and not always installed.
— Don't silence Ruff rules globally. If a rule is wrong for this codebase, discuss in an issue.
</dont>
</patterns>

<examples>
Idiomatic header for a new validator:

```python
#!/usr/bin/env python3
"""Validate <thing>. Runs inside `make check`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ROOT / "thing")
    args = parser.parse_args()
    # … checks …
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
</examples>

<troubleshooting>
| Symptom                                                          | Cause                                                | Fix                                                                          |
|------------------------------------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------|
| Ruff `I001 unsorted import`                                      | Import order                                         | `ruff check --fix`; if false-positive, add file to `[tool.ruff.lint.isort.known-first-party]` |
| `SIM`/`B` rule complains about pattern that is intentional        | Genuinely needed pattern                             | Add `# noqa: B017` with one-line rationale; do NOT globally disable          |
| `make py-lint` says "ruff not installed"                          | Ruff missing                                         | `uv pip install ruff` or `pip install ruff`; gate falls back to `py_compile` |
| `export_onnx.py` produces opset 18 file but Tract refuses it      | Tract wants opset 17, fixed-batch, `dynamo=False`     | See `python/export_onnx.py` Tract-compat branch; do not regress              |
| `cost_ledger.py check` parses 0 rows                              | Column order / header drift in `reports/cost.md`     | Compare against an existing row; restore exact column headings               |
| `decode_so100_to_h5.py` hangs on Ctrl-C                           | Not raising `SystemExit(130)`                        | Use the canonical `raise SystemExit(130) from None`                          |
</troubleshooting>

<references>
- `python/pyproject.toml` — Ruff config + deps
- `python/Makefile` — Python-side gate
- `scripts/check_*.py` — patterns for new validators
- `python/export_onnx.py` — non-trivial inference of `action_dim`; Tract-compat constraints
- `python/cost_ledger.py`, `python/hf_pricing.py` — cost discipline
- `CHANGELOG.md` "Polish: Ruff lint baseline" entry — historical context for the lint rollout
</references>
