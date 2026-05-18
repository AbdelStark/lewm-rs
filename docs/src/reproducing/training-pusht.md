# Reproducing PushT training

## 1. The hardware contract

The approval-gated PushT 50 k-step training job currently uses
**A10G-large** on Hugging Face Jobs, but the full Burn/Jepa path is
CPU-backed until the CUDA trainer is wired through `lewm-gpu`. The
previous bounded-core 50 k-step run took **5.3 hours** at \$1.50/hr,
totalling **\$7.95 USD**. The full Burn/Jepa job must be remeasured
before those numbers are treated as the release runtime.

Equivalent hardware (RTX 4090, RTX 6000 Ada, H100) should converge
faster but produce slightly different bit-level losses due to CUDA
kernel reduction-order differences.

## 2. The configs

```sh
cat configs/pusht.toml
```

The config locks all hyperparameters: peak LR 3e-4, final LR 1e-5,
warmup 1000, weight decay 0.05, batch 64, grad accum 2, 50 000 steps,
seed 0, AdamW $\beta_1=0.9, \beta_2=0.95$.

Override one or two via `--set KEY=VALUE` flags on the CLI, but for a
faithful reproduction leave them at the defaults.

## 3. Launch

```sh
scripts/launch_hf_job.py jobs/train_pusht.yaml --allow-approval-required
```

`jobs/train_pusht.yaml` declares:

- Image: `ghcr.io/abdelstark/lewm-rs:latest` (built from the checked-in
  `Dockerfile`).
- Hardware: A10G-large.
- Env: `HF_TOKEN`, `TRACKIO_PROJECT`, `TRACKIO_RUN`, and optional OTLP endpoint.
- Command:

  ```sh
  lewm-train train \
      --config configs/pusht.toml \
      --set 'experimental.pusht_train_mode="full_burn_jepa"' \
      --device cpu \
      --data-dir /tmp/data \
      --output-dir /tmp/out \
      --resume-if-present \
      --max-steps ${LEWM_MAX_STEPS:-50000}
  ```

The job uploads the trainer output directory to
`abdelstark/lewm-rs-pusht/train/pusht-full-burn-jepa-<UTC timestamp>/`
after the train command completes and after `python/export_onnx.py
--check-contract-only` verifies that the produced safetensors recovers the
exact 255 Burn destination tensors and all 303 PyTorch source keys required
by the ONNX export contract.

Before launching the paid job, run the local contract smoke:

```sh
scripts/full_pusht_contract_smoke.py \
  --report reports/full_pusht_contract_smoke.json
```

This runs one CPU step with `configs/pusht.toml` and validates the resulting
safetensors against the same 255-destination / 303-source ONNX export contract.
The optional report records the parsed `303/303` PyTorch-key recovery,
255 Burn destination tensors, and safetensors SHA-256. It does not upload
artifacts or launch a Hugging Face Job.

## 4. Monitoring

While the job is running:

- Stdout JSONL is captured by HF Jobs and exposed in the run UI.
- If `OTEL_EXPORTER_OTLP_ENDPOINT` is configured (in the job env),
  spans flow to your local Tempo/Grafana stack
  ([`infra/otel/`](https://github.com/AbdelStark/lewm-rs/blob/main/infra/otel/README.md)).

## 5. Resume on crash

The HF Jobs runner is robust to most failures. If a job dies mid-run:

Re-launch the same job after confirming the previous output directory is
still available. The trainer picks up at the last complete checkpoint via
the job spec's `--resume-if-present` flag.
The resume is bit-identical-state from the sidecar; see
[Determinism](../training/determinism.md).

## 6. After the run

The Hub repo `abdelstark/lewm-rs-pusht/train/<run_id>/` will contain:

- `step_0050000.{mpk,safetensors,json,parity.json}`
- `train_losses.jsonl`, `train_report.json`
- Intermediate checkpoints at `step_0005000` and step multiples.

## 7. Local-only reproduction (CPU smoke)

If you want a 100-step training run on CPU without HF Jobs:

```sh
cargo run --release -p lewm-train -- \
    --config configs/pusht.toml \
    --device cpu \
    --output-dir /tmp/lewm-train-pusht-cpu \
    --max-steps 100 train
```

Loss at step 100 should match the cloud BF16 run within TOL-005 (rel.
< 1e-2). This is a coarse sanity check, not a parity test, but it
catches gross plumbing bugs.

## 8. The cost ledger

Append your run to `reports/cost.md`:

```text
| 2026-05-15 | abdelstark/6a06f0c43308d79117b90276 | A10G-large | 5.3 h | $7.95 |
```

Run the cost-cap check:

```sh
python3 python/cost_ledger.py check --path reports/cost.md --cap-usd 200
```

If the check fails, you would have exceeded the project ceiling and
the run should not have launched.

## 9. Where to read

- Full report: [`reports/pusht_training.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/pusht_training.md).
- HF Jobs spec: [`jobs/train_pusht.yaml`](https://github.com/AbdelStark/lewm-rs/blob/main/jobs/train_pusht.yaml).
- Dockerfile: [`Dockerfile`](https://github.com/AbdelStark/lewm-rs/blob/main/Dockerfile).
