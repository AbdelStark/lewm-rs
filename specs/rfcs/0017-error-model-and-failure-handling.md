---
rfc: "0017"
title: "Error model, failure handling, panic policy, error message style"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: []
depends_on: ["0001"]
related: ["0004", "0005", "0010", "0016"]
---

# RFC 0017 — Error model, failure handling, panic policy, error message style

> **Status:** Accepted · **Version:** 1.0.0
>
> Error handling is the difference between "the model failed to converge" and "the trainer ran for 6 hours, swallowed an `unwrap`, wrote a corrupt checkpoint, then crashed at upload." This RFC pins the error type strategy, the panic policy, the error message style guide, and the recovery patterns.

---

## 1. Introduction

### 1.1 Motivation

Rust's `Result` is a powerful tool, but it requires discipline: too little structure and we hide signal; too much and we drown in boilerplate. We adopt the *libraries-use-thiserror, binaries-use-anyhow* convention and lock in a small style guide for error messages so the user-facing failure mode is consistent.

### 1.2 Goals

1. Pin the error type strategy per crate.
2. Pin the panic policy.
3. Pin the error message style guide (the three-part shape).
4. Specify recovery patterns for the trainer's outer loop.
5. Specify the conversion patterns between library errors and the binary's `anyhow::Error`.

### 1.3 Non-goals

- Comprehensive enumeration of error variants — each crate's RFC lists its own.

---

## 2. Conventions

- **Library error** — a domain-specific `enum` deriving `thiserror::Error`.
- **Binary error** — `anyhow::Error` at the outermost layer of a `main` function.
- **Invariant** — a condition the code expects to be true at a point; violation is a bug.
- **User error** — an error caused by user input or external state; not a bug.

---

## 3. Strategy

### 3.1 Libraries

Each library crate (`lewm-core`, `lewm-data`, `lewm-hub`, `lewm-telemetry`, `lewm-plan`, `lewm-infer`) defines an error enum:

```rust
// crates/<crate>/src/errors.rs

#[derive(thiserror::Error, Debug)]
pub enum LewmCoreError {
    #[error("invalid tensor shape: expected {expected:?}, found {found:?}")]
    InvalidShape { expected: Vec<usize>, found: Vec<usize> },

    #[error("module construction failed: {reason}")]
    ConstructionFailed { reason: String },

    #[error("parameter '{name}' not found in record")]
    ParamNotFound { name: String },

    #[error("{0}")]
    Other(String),
}
```

**RFC0017-001 [MUST]** — Every library crate exposes a single top-level error enum that is `Send + Sync + 'static`. Sub-modules may have private inner enums; the public surface is one.

**RFC0017-002 [MUST]** — Each variant carries enough structured data for programmatic handling (no string-only variants except `Other`).

**RFC0017-003 [MUST]** — `Other(String)` exists only as a catch-all for genuinely unforeseen cases. New occurrences trigger a refactor to a proper variant.

### 3.2 Binaries

Each binary's `main` returns `anyhow::Result<()>`. Errors propagate via `?` until they hit `main`, where they print using `{:#}` (which includes the chain).

```rust
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
    Ok(())
}
```

**RFC0017-004 [MUST]** — `anyhow::Context::context` is used to add caller context as errors propagate. Pattern:

```rust
let stats = compute_stats(&dataset)
    .context("computing training-split action statistics")?;
```

### 3.3 Conversion

Library errors implement `Into<anyhow::Error>` via the `thiserror`-generated `std::error::Error` impl. We do not write manual `From` impls between library errors except where natural (e.g., `DataError → TrainError`).

---

## 4. Panic policy

### 4.1 What may panic

Production code (non-test) **MAY** panic only for:

1. **Provably impossible invariant violations**. Tagged with `expect("invariant: <statement>")` where `<statement>` is a one-line natural-language assertion that this point is unreachable in any valid input.
2. **Allocator failure** — implicit via OOM aborts. We do not handle.
3. **Unrecoverable corruption** — e.g., a serialization library returning an internally inconsistent state.

### 4.2 What may not panic

- Any function consuming user input or external state (file path, env var, network response, dataset content) **MUST NOT** panic. It returns `Result`.
- Math operations that may divide by zero **MUST NOT** panic; check explicitly or use safe-div helpers.
- Indexing into a tensor whose shape is determined at run time **MUST NOT** panic; use `.get(..)` or check shape first.

**RFC0017-005 [MUST]** — `unwrap()` is denied by `clippy::unwrap_used = "deny"` in workspace lints. Exceptions: test code only.

**RFC0017-006 [MUST]** — `expect()` is warned by `clippy::expect_used`. Each call site **MUST** carry an `// invariant: ...` comment immediately above, explaining why the call cannot fail.

**RFC0017-007 [MUST]** — Release builds use `panic = "abort"`. Reason: the trainer's recovery model is resume-from-checkpoint, not unwinding from a panicking CUDA stream.

### 4.3 Panic captures

When a panic does happen (in test code or via abort), the trainer's signal handler tries to:

1. Write `panic.txt` with the panic message and a backtrace (if available).
2. Flush observability buffers.
3. Exit non-zero.

```rust
fn install_panic_hook(output_dir: PathBuf) {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!("{info}");
        let bt = std::backtrace::Backtrace::force_capture();
        let _ = std::fs::write(output_dir.join("panic.txt"), format!("{msg}\n\n{bt}"));
        default_hook(info);
    }));
}
```

---

## 5. Error message style guide

Every user-visible error message follows the **three-part shape**:

```
<what failed>: <smallest reproducer or context>; <fix or next step>
```

Examples:

```
invalid tensor shape: expected [B, T, 3, 224, 224], found [B, T, 3, 240, 320]; resize input to 224x224 or update config encoder.image_size
checkpoint not found: '/run/abc/step_0014400.mpk' (no such file); list checkpoints with `lewm-eval list /run/abc`
HF Hub upload failed: 401 Unauthorized for abdelstark/lewm-rs-pusht; verify HF_TOKEN scope includes write
```

**RFC0017-008 [MUST]** — The "what failed" part is a noun phrase, lowercase, no trailing punctuation.
**RFC0017-009 [MUST]** — The "context" part is concrete and minimal (file path, shape, status code).
**RFC0017-010 [MUST]** — The "fix" part is imperative and actionable.

### 5.1 The miette layer

For binaries we use `miette` to render errors with rich formatting and source highlighting where applicable:

```rust
fn main() -> miette::Result<()> {
    let cli = Cli::parse();
    run(cli).map_err(miette::Report::new_boxed)
}
```

`miette` renders fancy diagnostics in TTY mode and plain in piped output.

---

## 6. Recovery patterns

### 6.1 Trainer outer loop

```rust
fn run(cli: Cli) -> anyhow::Result<()> {
    let config = load_config(&cli.config).context("loading config")?;
    let mut state = State::Init;
    loop {
        let next = match state {
            State::Init => init_run(&cli, &config).context("INIT")?,
            State::ParityCheck => match run_parity(&config) {
                Ok(()) => State::Smoke,
                Err(ParityError::Tolerance(diff)) => {
                    bail!("parity check failed: L_inf = {diff:.4e}; check weight conversion and op fidelity");
                }
                Err(other) => return Err(other.into()),
            },
            State::Smoke => run_smoke(&config).context("SMOKE")?,
            ...
        };
        write_transition(&cli.output_dir, &state, &next)?;
        state = next;
        if state == State::Done { break; }
    }
    Ok(())
}
```

**Pattern:** state-machine + explicit error-to-next-state mapping where recovery is meaningful (e.g., `ParityError::Tolerance` is fatal; `IoError::Interrupted` may be retriable).

### 6.2 NaN / Inf in gradient

Pattern in [RFC 0005 §5.2](0005-training-system.md): three skipped steps → fatal. The fatal path emits a structured error and writes an artifact.

### 6.3 Network errors

Pattern in [RFC 0010 §5.2](0010-huggingface-hub-integration.md): exponential backoff. After max retries, surface a `HubError::Timeout` with the partial-progress summary.

### 6.4 Data errors

Pattern in [RFC 0004 §10](0004-data-pipeline.md): one transient error per shard is recoverable (log + skip index); two consecutive on the same shard is fatal.

---

## 7. Cross-crate error propagation

```
lewm-data::DataError ─────► lewm-train::TrainError::Data(DataError)
lewm-hub::HubError    ─────► lewm-train::TrainError::Hub(HubError)
lewm-core::LewmCoreError ──► lewm-train::TrainError::Core(LewmCoreError)
                              │
                              ▼  ?  (anyhow::Error::msg or thiserror's #[from])
                            main's anyhow::Result
```

The `TrainError` enum:

```rust
#[derive(thiserror::Error, Debug)]
pub enum TrainError {
    #[error(transparent)]
    Data(#[from] DataError),
    #[error(transparent)]
    Core(#[from] LewmCoreError),
    #[error(transparent)]
    Hub(#[from] HubError),
    #[error(transparent)]
    Telemetry(#[from] TelemetryError),
    #[error("config error: {0}")]
    Config(String),
    #[error("parity failed at {state}: {detail}")]
    Parity { state: String, detail: String },
    #[error("smoke failed: {0}")]
    Smoke(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error("optim error: {0}")]
    Optim(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("internal: {0}")]
    Internal(String),
}
```

**RFC0017-011 [MUST]** — Lower-crate errors are `#[error(transparent)]` (delegate Display to inner). Crate-specific errors carry an explicit message format.

---

## 8. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0017-MSG-001 | `error_messages_three_part_shape` | unit | §5 |
| TST-0017-MSG-002 | `display_chains_with_context` | unit | §3.2 |
| TST-0017-PANIC-001 | `panic_in_release_aborts` | integration | RFC0017-007 |
| TST-0017-PANIC-002 | `panic_writes_panic_txt` | integration | §4.3 |
| TST-0017-UNWRAP-001 | `no_unwrap_in_production_code` | meta (clippy CI) | RFC0017-005 |

### 8.1 Style enforcement

A small `scripts/lint_errors.py` lint regexes every `#[error(...)]` attribute and ensures the format matches the three-part shape. It is informational (warning) rather than blocking, because not every error string fits perfectly; the human reviewer is final arbiter.

---

## 9. Operational considerations

### 9.1 Observability

- `error/<kind>` counter incremented per occurrence (e.g., `error/data/io_wait_timeout`).
- Every panic captured by hook is also emitted as a CRITICAL log line.

### 9.2 Runbook

- **"User reports a confusing error message."** — file an issue with the `error-uX` label; we treat error UX as a first-class concern.

---

## 10. Performance considerations

`thiserror` derives are zero-cost. `anyhow::Error` allocations are minimal. No hot-path concerns.

---

## 11. Security considerations

- Error messages **MUST NOT** include secrets. The redactor (RFC 0009 §7.4) handles tracing-emitted strings.
- Backtraces in `panic.txt` may include file paths from the build environment; these are sanitized via the `--remap-path-prefix` flag (RFC 0011 §8).

---

## 12. Alternatives considered

- **A1 — `eyre` over `anyhow`.** Considered; `anyhow` is more widely known and adequate.
- **A2 — All errors via `Box<dyn Error>`.** Rejected: loses programmatic handling.
- **A3 — `snafu` over `thiserror`.** `thiserror` is the de-facto standard; we follow.

---

## 13. Acceptance criteria

- [ ] All TST-0017-* pass.
- [ ] Every library crate has an error enum.
- [ ] CI lint catches `unwrap()` in production code.
- [ ] Trainer's panic hook writes `panic.txt`.

---

## 14. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Error enum sprawl | M | L | Other variant explicitly forbidden except as last resort |
| R-2 | Error message drift | M | L | Style guide; informational lint |
| R-3 | Backtrace bloat in panic.txt | L | L | `--remap-path-prefix` |

---

## 15. Open questions

OQ-2017-1 — Whether to embed error documentation in the rustdoc of each `Err` variant (so users see them by clicking). Default yes; CI lint to enforce later if needed.

---

## 16. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0017.*
