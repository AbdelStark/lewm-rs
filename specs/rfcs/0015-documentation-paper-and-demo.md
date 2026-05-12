---
rfc: "0015"
title: "Documentation, paper writeup, demo Space"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§3 D5/D7/D8", "§7.2 P13", "§8 Phase 7"]
depends_on: ["0001", "0007", "0010", "0011"]
related: ["0009"]
---

# RFC 0015 — Documentation, paper writeup, demo Space

> **Status:** Accepted · **Version:** 1.0.0
>
> The technical work is half the deliverable; the other half is how it is **explained**. This RFC pins the rustdoc style, the paper-style writeup structure, the README, the Gradio Space contents, and the blog post.

---

## 1. Introduction

### 1.1 Motivation

A reproducible result that no one can find or follow has limited value. This RFC turns the artifacts into a coherent narrative: README → paper → blog → demo, with rustdoc as the source-of-truth API reference.

### 1.2 Goals

1. Specify rustdoc conventions enforced by CI.
2. Specify the paper writeup outline and its acceptance gate.
3. Specify the README contents and its required sections.
4. Specify the Gradio Space layout and behaviour (the demo).
5. Specify the blog post outline.

### 1.3 Non-goals

- The model cards (covered by [RFC 0010 §7](0010-huggingface-hub-integration.md)).
- The training/eval reports (covered by [RFC 0005](0005-training-system.md) and [RFC 0006](0006-planning-and-evaluation.md)).

---

## 2. Conventions

- Markdown is GFM (CommonMark + tables + fenced code).
- Paper is rendered to PDF via `pandoc` with the `eisvogel` template.
- Documentation pages live on GitHub Pages at `https://abdelstark.github.io/lewm-rs/`.

---

## 3. Rustdoc

### 3.1 Required documentation

Every public item in every crate **MUST** carry a documentation comment. Enforced by the workspace lint `missing_docs = "warn"` + CI `RUSTDOCFLAGS="-D warnings"`.

**RFC0015-001 [MUST]** — Module-level docs at the top of every `lib.rs` and `mod.rs` describe the module's purpose in one paragraph and link to the relevant RFC.

**RFC0015-002 [MUST]** — Public functions document:

- A one-line summary.
- A `# Shape` section for any function consuming or producing tensors, with input and output shape lines.
- An `# Errors` section enumerating possible error variants.
- An `# Invariants` section for any function with non-obvious preconditions/postconditions.
- A `# Example` section for any function that is part of the user-facing API.

### 3.2 Style

```rust
/// Encode a windowed image tensor to embeddings.
///
/// # Shape
/// - input  `pixels: (B, T, C, H, W)`
/// - output `(B, T, D)`
///
/// # Invariants
/// - `C == self.config.encoder.num_channels`
/// - `H == W == self.config.encoder.image_size`
///
/// # Errors
/// Returns `LewmCoreError::InvalidShape` if any invariant is violated.
///
/// # Example
/// ```ignore
/// let z = jepa.encode(pixels);
/// ```
///
/// # See also
/// - [RFC 0002 §4.8.3](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md#483-encode)
pub fn encode(&self, pixels: Tensor<B, 5>) -> Tensor<B, 3> { /* … */ }
```

**RFC0015-003 [SHOULD]** — Doc examples are marked `ignore` if they require GPU; tested as `### Tests` blocks otherwise.

**RFC0015-004 [MUST]** — Every doc cross-link to an RFC uses a stable URL (linking to a section anchor in the GitHub-rendered markdown).

### 3.3 Module index

Each crate's `lib.rs` ends with a module index for navigation:

```rust
//! ## Module index
//!
//! - [`vit`] — Vision Transformer encoder. See [RFC 0002 §4.2](...).
//! - [`predictor`] — Autoregressive predictor with AdaLN-zero. See [RFC 0002 §4.7](...).
//! - [`losses`] — SIGReg and prediction loss. See [RFC 0003](...).
```

---

## 4. README

### 4.1 Structure (top-level `README.md`)

```markdown
# lewm-rs

> Pure-Rust reproduction and extension of LeWorldModel (Maes et al., 2026).

[ Badges: build status, license, crates.io, docs.rs, model on HF, dataset on HF, demo Space ]

## What

One-paragraph plain-English summary. What it is, what it does, why it matters.

## Quickstart

A self-contained 5-line code block that does *something* visible.

## Results

A small table of the headline metrics with links to model cards and reports.

## Architecture at a glance

A single ASCII or Mermaid diagram of the system.

## Reproducing

A 3-bullet recipe: clone, run X, run Y. Detailed runbook lives in RFC 0005 §9.

## Project structure

A tree of the top-level directories with a one-line description each.

## License

MIT for code; Apache-2.0 for checkpoints; CC-BY-4.0 for the writeup.

## Citation

bibtex block.

## Acknowledgments

LeCun, Maes, Le Lidec, Scieur, Balestriero (LeWM authors); Lucas-Maes for the
upstream reference code; the Burn team for the framework.
```

**RFC0015-005 [MUST]** — README **MUST NOT** exceed 200 lines. Detail belongs in the docs or RFCs.

**RFC0015-006 [MUST]** — README **MUST** link to the spec set (`specs/`) and to the most recent model and Space.

### 4.2 Sub-READMEs

Each crate has its own `README.md` with:

- One-paragraph purpose.
- Public API surface (rendered from the rustdoc into `src/lib.rs` re-exports).
- A link to the owning RFC.

---

## 5. Paper writeup

### 5.1 Structure (`paper/lewm-rs.md`)

```markdown
# lewm-rs: A Pure-Rust Reproduction of LeWorldModel

Abdel · 2026

## Abstract

~ 200 words. The result, the method, the artifact.

## 1. Introduction

Why this work. What LeWM is. What "pure Rust" adds. What our contributions are.

## 2. Background

### 2.1 JEPA and LeWM
### 2.2 The Burn / Tract Rust ML stack
### 2.3 The SO-100 dataset

## 3. Architecture

### 3.1 Encoder
### 3.2 Predictor with AdaLN-zero
### 3.3 Action encoder
### 3.4 SIGReg

(Each subsection cites the implementing RFC.)

## 4. Training pipeline

### 4.1 Data plane
### 4.2 Optimizer and schedule
### 4.3 Mixed precision
### 4.4 Determinism contract
### 4.5 Observability

## 5. Parity testing

What we test, the tolerances, the result.

## 6. PushT result

### 6.1 Training curves
### 6.2 Eval: planning success rate
### 6.3 λ sweep ablation
### 6.4 Cost ledger

## 7. SO-100 extension

### 7.1 Dataset preparation
### 7.2 Warm-start vs scratch
### 7.3 Eval: latent rollout, Spearman
### 7.4 Discussion

## 8. CPU inference

### 8.1 ONNX export pipeline
### 8.2 Latency on laptop and CPU XL
### 8.3 Demo Space

## 9. Lessons learned

What surprised us. What didn't work. What's worth revisiting.

## 10. Related work

Brief, not exhaustive.

## 11. Future work

What v2 might do.

## 12. Conclusion

## Appendix A: full hyperparameter table
## Appendix B: per-layer parameter count
## Appendix C: reproducibility checklist

## Acknowledgments

## References
```

### 5.2 Style

- Plain prose; no marketing fluff.
- Every claim is backed by a number from the reports or a link to the RFC.
- Figures generated from CSV / parquet via `python/plot_curves.py` and committed under `paper/figures/`.
- BibTeX in `paper/bibliography.bib`.

**RFC0015-007 [MUST]** — All numerical claims in the paper **MUST** be sourced from `reports/*.md`. CI verifies the link integrity.

**RFC0015-008 [SHOULD]** — Paper length 8–12 pages PDF (post-render). Not a hard rule; over-long writeups are not better.

### 5.3 Rendering

```bash
pandoc paper/lewm-rs.md \
    --template eisvogel \
    --listings \
    -o paper/lewm-rs.pdf \
    --bibliography paper/bibliography.bib \
    --citeproc
```

`docs.yml` runs this on every push to `main`; the PDF lives at `https://abdelstark.github.io/lewm-rs/paper/lewm-rs.pdf`.

---

## 6. Demo Space (`AbdelStark/lewm-rs-demo`)

### 6.1 UI

A Gradio interface with:

```
┌─────────────────────────────────────────────────────────┐
│ Inputs                                                  │
│  ▢ Start image upload                                   │
│  ▢ Goal image upload                                    │
│  ▢ Horizon (slider, 3–8, default 5)                     │
│  ▢ N candidates (slider, 8/16/32, default 16)            │
│  [ Submit ]                                              │
├─────────────────────────────────────────────────────────┤
│ Outputs                                                  │
│  ▸ Predicted cost: 0.42                                 │
│  ▸ Action sequence: [0.12, -0.04, 0.31, ...]            │
│  ▸ Latency: 0.87 s                                       │
│  ▸ ASCII action trajectory:                              │
│       step 0:  ↗                                         │
│       step 1:  →                                          │
│       step 2:  →                                          │
│       step 3:  ↘                                          │
│       step 4:  ↓                                          │
└─────────────────────────────────────────────────────────┘
```

Plus a sidebar:

- About this demo (paragraph).
- Link to paper.
- Link to source code.
- Link to model card.

### 6.2 Implementation

```python
# space/app.py
import gradio as gr
import subprocess, json

def plan(start, goal, horizon, n_cand):
    result = subprocess.run(
        ["/usr/local/bin/lewm-infer", "plan",
         "--checkpoint-dir", "/ckpt",
         "--start", start, "--goal", goal,
         "--horizon", str(horizon), "--n-cand", str(n_cand),
         "--out", "/tmp/out.json"],
        check=True, capture_output=True, timeout=30,
    )
    with open("/tmp/out.json") as f:
        data = json.load(f)
    return data["cost"], json.dumps(data["best_actions"]), data["latency_ms"]

with gr.Blocks() as demo:
    gr.Markdown("# lewm-rs CPU planning demo")
    with gr.Row():
        with gr.Column():
            start = gr.Image(type="filepath", label="Start")
            goal = gr.Image(type="filepath", label="Goal")
            horizon = gr.Slider(3, 8, 5)
            n_cand = gr.Slider(8, 32, 16, step=8)
            btn = gr.Button("Plan")
        with gr.Column():
            cost = gr.Number(label="Cost")
            actions = gr.Textbox(label="Best actions")
            latency = gr.Number(label="Latency (ms)")
    btn.click(plan, [start, goal, horizon, n_cand], [cost, actions, latency])

demo.queue(concurrency_count=1).launch()
```

**RFC0015-009 [MUST]** — The Space `requirements.txt` is **minimal** (gradio + nothing else). All ML work happens in the Rust binary.

**RFC0015-010 [MUST]** — Auto-pause after **15 minutes** of inactivity (HF Space setting).

**RFC0015-011 [MUST]** — Example images: two pre-uploaded start/goal pairs (PushT and SO-100) to help first-time users.

### 6.3 Space card

`space/README.md` is a small card explaining the demo, with the standard YAML frontmatter required by HF Spaces:

```yaml
---
title: lewm-rs CPU planning demo
emoji: 🤖
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: mit
---
```

---

## 7. Blog post

### 7.1 Outline (Hub blog post)

```markdown
# Building lewm-rs: A Pure-Rust JEPA World Model

Audience: ML engineers and Rust developers curious about post-PyTorch tooling.
Length: ~ 1500 words.
Tone: technical but accessible.

## Section 1 — The pitch (200 words)
What LeWM is. Why a Rust port is interesting. The artifacts.

## Section 2 — The Rust stack (300 words)
Burn for training. Tract for inference. What hurt, what worked.

## Section 3 — Parity testing (300 words)
The single highest-value local check before any cloud spend.

## Section 4 — Cost engineering (300 words)
The 80–100 USD budget; how we kept inside it. The ml-intern leash.

## Section 5 — The SO-100 extension (300 words)
Warm-start; the null-result discipline.

## Section 6 — What's next (100 words)
v2 ideas. Verifiable AI angle.

## Section 7 — Try it (links)
Demo Space; model repos; source code; paper.
```

**RFC0015-012 [MUST]** — Blog post links to the paper PDF, the demo Space, the source repo, the model cards.

---

## 8. Documentation hosting

GitHub Pages at `https://abdelstark.github.io/lewm-rs/` serves:

```
/                       # rendered README + nav
/docs/                  # cargo doc output
/specs/                 # mdbook of the spec set
/paper/                 # the PDF
/reports/               # the rendered training/eval reports
```

`docs.yml` (RFC 0011 §3.5) builds and publishes on every push to `main`.

---

## 9. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0015-DOC-001 | `cargo_doc_warning_free` | unit (CI) | RFC0015-001/002 |
| TST-0015-LINKS-001 | `paper_internal_links_valid` | python | RFC0015-007 |
| TST-0015-PAPER-001 | `paper_pdf_renders_without_errors` | integration | §5.3 |
| TST-0015-PAPER-002 | `paper_has_all_required_sections` | python | §5.1 |
| TST-0015-BLOG-001 | `blog_post_renders_and_links_valid` | python | §7 |
| TST-0015-DEMO-001 | `demo_space_app_smoke` | integration | §6.2 |
| TST-0015-README-001 | `readme_under_200_lines` | unit | RFC0015-005 |
| TST-0015-README-002 | `readme_has_required_sections` | unit | §4.1 |

---

## 10. Operational considerations

### 10.1 Runbook

- **"Paper PDF render fails."** — almost certainly a LaTeX-citation issue; `pandoc --verbose` reveals it. Common cause: missing bibtex entry.
- **"Space is paused but user clicked through."** — Gradio shows a "loading" message during cold start (~ 30 s). Documented in Space card.

### 10.2 Capacity

Pages site is essentially free. Space at T4 small ≤ 15 USD/month per PRD §7.2.

---

## 11. Performance considerations

None beyond inference latency (RFC 0014 §4).

---

## 12. Security considerations

- Space inputs (user-uploaded images) flow into a sandboxed Rust binary; safe.
- No PII collected.

---

## 13. Alternatives considered

- **A1 — Single big README, no paper.** Rejected: the project's research framing requires a paper-style writeup.
- **A2 — Jupyter notebook as docs.** Rejected; we ship Rust binaries, not notebooks.

---

## 14. Acceptance criteria

- [ ] All TST-0015-* pass.
- [ ] Paper PDF on Pages.
- [ ] Space live.
- [ ] Blog post published.

---

## 15. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Paper takes longer than 1 week to write | M | L | Outline locked here; numbers come from reports |
| R-2 | Space cost overrun | L | L | Auto-pause; CPU fallback |
| R-3 | Gradio SDK version drift | M | L | Pin in `requirements.txt`; CI smoke |

---

## 16. Open questions

OQ-2015-1 — Whether to submit the paper to arXiv. Defer; PRD §3 says "suitable for arXiv tech report" — submission decision after Phase 7.

---

## 17. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0015.*
