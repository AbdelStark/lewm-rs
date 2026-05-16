# ONNX export pipeline

> **Motivation.** The deployment story is *Tract CPU inference*. To
> get there from a Burn-trained checkpoint we cross two compatibility
> boundaries (Burn → Safetensors → PyTorch → ONNX → Tract). This page
> documents that pipeline and the quirks at each boundary.
>
> **Position.** Top of [Part V — Inference and deployment](./onnx-export.md).
>
> **What you should leave with.** The two ONNX variants (`dynamo=True`
> onnxruntime-compat and `dynamo=False` Tract-compat), the opset
> choices, and the four gotchas that broke Tract until they were fixed.

## 1. Two graphs, not one fused graph

The first design choice: we export **two separate ONNX graphs**, not
a single fused JEPA graph.

- **Encoder graph** — `(B=1, 3, 224, 224) f32 → (1, 192) f32`. Maps
  pixels to the CLS embedding.
- **Predictor graph** — `(B, T, 192) f32, (B, T, A) f32 → (B, T, 192) f32`.
  Maps history + actions to the predicted latent.

The reason: at planning time, CEM batches the *predictor* over
`n_cand = 1024` candidates per iteration but invokes the *encoder*
only twice per decision (once on the current observation, once on the
goal). Exporting as two graphs lets the planner shape the batch axis
independently for each.

## 2. The two export variants

| Variant | onnxruntime / Gradio Space | Tract CPU runner |
|---------|----------------------------|------------------|
| Exporter | dynamo (`torch.onnx.export(..., dynamo=True)`) | legacy TorchScript (`dynamo=False`) |
| Opset | 18 | 17 |
| Batch axis | dynamic (`dynamic_axes={0: "batch"}`) | fixed (`batch=1`) |
| Symbolic shapes | yes | no |
| Files | `encoder.onnx` (378 KB metadata) + `encoder.onnx.data` (25 MB weights) and similarly for predictor | `encoder.onnx` (25 MB, embedded weights) + similarly for predictor |
| Stored under | `<repo>/...` root | `<repo>/tract-compat/...` |

Why two variants? **Tract 0.22.1 does not parse all opset-18
constructs**, in particular some symbolic-shape operations the dynamo
exporter produces (e.g. `Min(3, history_dim)`-style annotations).
Reverting to the legacy TorchScript exporter with opset 17 and fixed
batch size produces a graph that Tract handles cleanly.

The Gradio demo Space uses the onnxruntime-compat variant because
onnxruntime *does* support opset 18 dynamo output and benefits from
the dynamic batch axis.

## 3. The four Tract-compat gotchas

In addition to the opset / exporter choice, four implementation details
must be right for Tract to parse the graph:

### 3.1 No `dynamic_axes`

Setting any `dynamic_axes` produces symbolic shape annotations in the
exported graph that Tract's shape inference rejects (`InferenceConcat`
failures). The Tract-compat variant uses fixed shapes throughout.

### 3.2 Causal mask as a buffer, not a built-at-forward op

The predictor's causal mask must be **pre-registered** as an
`nn.Module` buffer in `__init__`, not built inside `forward()`:

```python
# WRONG — produces dynamic torch.ones(T, T) in the ONNX graph
def forward(self, latents, actions):
    T = latents.shape[1]
    mask = torch.ones(T, T, device=latents.device).triu(1).bool()
    ...

# RIGHT — fixed-T mask, registered once
def __init__(self, T=3):
    ...
    self.register_buffer("causal_mask",
                        torch.ones(T, T).triu(1).bool(),
                        persistent=False)

def forward(self, latents, actions):
    mask = self.causal_mask
    ...
```

The wrong form produces a graph with a `OneLike` op fed by `Shape`,
which Tract cannot trace through. The right form bakes the constant
mask into the graph at export time.

### 3.3 Action-dim inference from the smoother

`python/export_onnx.py` infers the encoder's expected action dim
**from the Conv1d smoother's weight shape**, not from a hardcoded
constant:

```python
action_dim = state_dict["action_enc.smoother.weight"].shape[1]
```

This recovers the encoder's `input_dim`: 10 for PushT (frameskip-packed
2-DOF actions) and 6 for SO-100 (raw 6-DOF actions). The recorded value
is what the predictor ONNX graph expects at runtime, so downstream
runners can size their action buffer without consulting the config.

### 3.4 Atomic per-arm export

Encoder and predictor are exported separately, each as its own
`torch.onnx.export` call. This isolates failures: if (say) a future
change breaks the predictor export, the encoder graph still ships.

## 4. The export pipeline end-to-end

```sh
python python/export_onnx.py \
    --checkpoint abdelstark/lewm-rs-pusht/train/.../step_0050000.safetensors \
    --variant tract-compat \
    --out-dir abdelstark/lewm-rs-pusht/tract-compat/
```

The script:

1. Loads the Safetensors checkpoint via `safetensors.torch.load_file`.
2. Builds the PyTorch reference model (`LeWMReference` class in
   `python/export_onnx.py`) and loads the state dict.
3. Sets the model to `.eval()`.
4. Constructs dummy inputs: `pixels (1, 3, 224, 224)`, `history (1, 3,
   192)`, `actions (1, 3, A)`, all on CPU.
5. Calls `torch.onnx.export` once for the encoder and once for the
   predictor, with the variant-specific flags.
6. Optionally runs a parity check: load the ONNX graphs via
   `onnxruntime.InferenceSession`, run on the same dummy inputs, and
   verify $L_\infty < 10^{-4}$ against the PyTorch forward.

## 5. The parity check

The ONNX-to-PyTorch parity check is part of the
[`reports/inference.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/inference.md)
output. It establishes that the ONNX graphs are numerically faithful
to the PyTorch model from which they were exported.

A separate check — `lewm-infer eval --dumps-dir ...` — runs the Tract
graph on the parity fixture and compares to the official reference
dumps. This is the "end-to-end" parity: Burn-train → Safetensors →
PyTorch ref → ONNX → Tract → activations vs upstream reference.

## 6. Source pointers

| Topic | Source |
|-------|--------|
| Exporter | `python/export_onnx.py` |
| Reference PyTorch model | `LeWMReference` class in `python/export_onnx.py` |
| Tract runner | `crates/lewm-infer/src/runner/tract_onnx_runner.rs` |
| Parity eval | `crates/lewm-infer/src/eval.rs` |
| Inference report | `reports/inference.md`, `reports/gpu_inference.md` |
