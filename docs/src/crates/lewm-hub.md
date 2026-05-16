# `lewm-hub`

Hugging Face Hub upload helpers. Used by `lewm-train` at the UPLOAD
state to push final artifacts.

## What it owns

- **Upload client**: a thin wrapper around `hf` CLI calls plus the
  Python helper `python/upload_checkpoints.py`.
- **Model card generation**: builds the README for each artifact repo
  from the training report and the model metadata.
- **Atomic upload sequencing**: ensures checkpoint files land in the
  Hub repo in a single, consistent batch.

## Public API

```rust,ignore
pub trait HubUploader {
    fn upload_checkpoint(&self, src_dir: &Path, dst_repo: &str, path_prefix: &str)
        -> Result<UploadReport, HubError>;
}
```

## Dependencies

- `reqwest` (for direct API calls)
- `tokio` (async runtime)
- `serde_json`
- Python helpers invoked via `std::process::Command` for HF CLI tasks
  that are easier in Python.

## Source

[`crates/lewm-hub`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-hub)
