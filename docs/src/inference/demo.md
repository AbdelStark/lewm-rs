# The Hugging Face demo Space

> **Motivation.** A live demo is the most direct way to convey what the
> model does. The Space hosts a Gradio app that, given a start image
> and a goal image, runs CEM and shows the planned action sequence.
>
> **Position.** Sub-page of [Part V](./onnx-export.md).
>
> **What you should leave with.** What the Space does, what it loads,
> and where the cost lives.

## 1. The Space

Live URL: [`abdelstark/lewm-rs-demo`](https://huggingface.co/spaces/abdelstark/lewm-rs-demo).

The Space is a Gradio app written in Python. At startup it:

1. Downloads the onnxruntime-compat ONNX graphs from
   [`abdelstark/lewm-rs-pusht`](https://huggingface.co/abdelstark/lewm-rs-pusht):
   `encoder.onnx`, `encoder.onnx.data`, `predictor.onnx`,
   `predictor.onnx.data`, and `stats.safetensors`.
2. Constructs onnxruntime `InferenceSession`s for the encoder and
   predictor.
3. Auto-detects the action dim from the predictor's input shape (so
   the Space works with both PushT and SO-100 checkpoints).
4. Starts the Gradio interface.

## 2. The user flow

```text
   User uploads:
       start_image  (any aspect ratio; resized to 224×224)
       goal_image   (same)

   App:
       z_history = stack of 3 copies of encoder(start_image)
       z_goal    = encoder(goal_image)
       a_1:H     = CEM(z_history, z_goal, predictor)
       (or the user can choose CEM iterations / candidates)

   Output:
       a_1:H displayed as a numeric table and a 2-D plot
       wall-time latency
```

The app does *not* run a physical robot or a simulator. The output is
purely the planner's action sequence. The user can interpret it as
"what the policy would do if asked to push the block from this
configuration to that one".

## 3. The ONNX session config

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

encoder_session = ort.InferenceSession("encoder.onnx", sess_options)
predictor_session = ort.InferenceSession("predictor.onnx", sess_options)
```

Note that the Space uses `onnxruntime`, not Tract — the dynamo-exported
graphs are not Tract-compatible (see [ONNX export](./onnx-export.md)
§2). The Space is the only inference path that uses onnxruntime; all
in-Rust paths use Tract.

## 4. Latency expectations

On Hugging Face Spaces' free CPU tier (basic AMD instance), CEM with
the default `n_iter = 5, n_cand = 1024` runs in roughly 6–10 seconds
per planning decision. This is slower than the local Apple M3 number
(4.08 s) because the Space's CPU is more constrained, but the order
of magnitude is consistent.

For the Space we typically expose lower defaults
(`n_iter = 3, n_cand = 512`) to keep the user-facing latency under
3 seconds.

## 5. Source pointers

| Topic | Source |
|-------|--------|
| Space app | Hosted in the `abdelstark/lewm-rs-demo` Space repo |
| ONNX assets | `abdelstark/lewm-rs-pusht` (root) |
| Stats | `stats.safetensors` |
| Action dim auto-detect | `python/export_onnx.py::infer_action_dim` |
