# Conformance Inputs

`make accept` looks for `conformance/hub_artifacts.json` and verifies every listed artifact with `scripts/check_hub_artifacts.py`. The manifest is intentionally absent until the Hub publication milestone provides stable artifact hashes.

Manifest shape:

```json
{
  "artifacts": [
    {
      "name": "PushT model card",
      "repo": "abdelstark/lewm-rs-pusht",
      "repo_type": "model",
      "revision": "main",
      "path": "README.md",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

`repo_type` may be `model`, `dataset`, or `space`. A direct `url` may be used instead of `repo`, `repo_type`, `revision`, and `path` for non-Hub fixtures or local validation.
