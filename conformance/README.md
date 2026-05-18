# Conformance Inputs

`make check` validates `conformance/release_blockers.json` with
`scripts/check_release_blockers.py --allow-open`. `make accept` runs the same
validator without `--allow-open`, so release acceptance fails while any listed
blocker has status `blocked`, `pending`, or `open`.

`make accept` also looks for `conformance/hub_artifacts.json` and verifies every
listed artifact with `scripts/check_hub_artifacts.py`. The manifest is
intentionally absent until the Hub publication milestone provides stable
artifact hashes.

The release blocker manifest is intentionally a complete mirror of the
production backlog: F1 through F13 must all be present and must map to GitHub
issues #243 through #255. The checker rejects missing, duplicate, unexpected,
or mis-numbered blocker IDs so the release gate cannot pass by silently
dropping a later backlog item.

Every `evidence` entry must be a repo-relative path that exists in the
checkout. Use `required_resolution` for future artifacts or live URLs that do
not exist yet.

Release blocker shape:

```json
{
  "schema_version": "1.0.0",
  "updated": "2026-05-18",
  "blockers": [
    {
      "id": "F1",
      "issue": 243,
      "phase": "A",
      "title": "Export trained full PushT ONNX artifacts",
      "status": "blocked",
      "evidence": ["reports/pusht_onnx_export.md"],
      "required_resolution": ["Upload verified onnx-full artifacts."]
    }
  ]
}
```

Use `status: "resolved"` only after the linked evidence proves the blocker is
actually complete.

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
