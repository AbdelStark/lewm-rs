# Reproducing SO-100 training

## 1. Hardware

A10G-large on HF Jobs, 5 000 steps, ~14 minutes, ~\$0.36 at \$1.50/hr.

## 2. Dataset preparation

The raw SO-100 dataset (`lerobot/svla_so100_pickplace`) stores frames
as AV1 video inside Parquet files. We first decode to HDF5 at 10 fps,
$224 \times 224$ pixels:

```sh
python python/decode_so100_to_h5.py \
    --src lerobot/svla_so100_pickplace \
    --dst abdelstark/so100-pickplace-lewm-ready \
    --fps 10 --image-size 224
```

The processed dataset is **1.9 GB / 6 559 timesteps / 50 episodes**.

## 3. Compute action stats

```sh
python python/compute_so100_stats.py \
    --src abdelstark/so100-pickplace-lewm-ready \
    --out stats.safetensors
```

`stats.safetensors` is appended to the dataset and embedded in the
training config.

## 4. Train

```sh
scripts/launch_hf_job.py jobs/train_so100.yaml --allow-approval-required
```

`jobs/train_so100.yaml` uses `configs/so100.toml`, which differs from
`configs/pusht.toml` in:

- `action_dim = 6` (raw 6-DOF joints).
- `max_steps = 5000`.
- `warmup_steps = 500`.
- `weight_decay = 0.01`.
- Dataset = `abdelstark/so100-pickplace-lewm-ready`.

All other hyperparameters (peak LR 3e-4, final LR 1e-5, batch 64,
$\beta$, AdaLN-zero defaults, SIGReg defaults) match PushT.

## 5. Warm-start variant

```sh
LEWM_PUSHT_WARMSTART_MPK=train/<run>/step_<N>.mpk \
scripts/launch_hf_job.py jobs/train_so100_warmstart.yaml --allow-approval-required
```

This job fails closed unless `LEWM_PUSHT_WARMSTART_MPK` points at a
compatible current bounded-core PushT `.mpk` source checkpoint. Full
Burn/Jepa `NamedMpk` records are not accepted by this bounded-core
SO-100 warm-start path. The job loads shared PushT modules into the
encoder, projector, predictor, and pred-proj before training, then
trains for 5 000 steps on SO-100. The action encoder is freshly
initialised because SO-100 has a different action input count.

The two runs (from-scratch and warm-start) can be compared via the
SO-100 eval CLI. See [Warm-start ablation](../planning/warm-start.md).

## 6. Local CPU smoke

```sh
cargo run --release -p lewm-train -- \
    --config configs/so100.toml \
    --device cpu \
    --output-dir /tmp/lewm-so100-cpu \
    --max-steps 100 train
```

Step-100 loss should be close to the cloud BF16 result.

## 7. Reports

- Full report: [`reports/so100_training.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/so100_training.md).
- Job spec: [`jobs/train_so100.yaml`](https://github.com/AbdelStark/lewm-rs/blob/main/jobs/train_so100.yaml).
