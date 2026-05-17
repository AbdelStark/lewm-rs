# Release runbook

This document is the **operator runbook** for cutting a `lewm-rs` release. It
captures the manual steps that are not automated and the verification gates
that must pass before a tag is promoted to a publication.

The end-to-end pipeline is implemented in
[`.github/workflows/release.yml`](.github/workflows/release.yml) and is wired
to the canonical `v*.*.*` git tags. The workflow runs unattended once a tag is
pushed; the human gate is the **draft → publish** promotion at the very end.

---

## 1. Versioning policy

* **Semantic versioning**, anchored to the API of the `lewm-train`,
  `lewm-infer`, and `lewm-plan` binaries plus the `lewm-core` library.
* **Major** (`v1.x.x`) — breaking changes to the spec contracts (RFC-bound),
  the CLI surface, the on-disk checkpoint layout, or the Hub artifact tree.
  Requires an ADR superseding the affected RFC clauses.
* **Minor** (`v0.x.0`) — additive behaviour: new training paths, new infer
  backends, new evaluation suites, new RFC-Accepted features.
* **Patch** (`v0.0.x`) — bug fixes, performance work, dependency updates that
  pass the audit/deny waivers, documentation refreshes.
* **Pre-releases** (`v0.x.y-rcN`) — opt-in via the same tag format with the
  `-rcN` suffix; the workflow short-circuits the Hub retag step.

The published version is mirrored in:

* `[workspace.package].version` in [`Cargo.toml`](Cargo.toml)
* Per-crate `[package].version` (kept in sync via `workspace = true`)
* The `[Unreleased]` heading in [`CHANGELOG.md`](CHANGELOG.md), promoted to
  the new version when cutting a release.

---

## 2. Pre-flight checklist

Run from a clean checkout of `main` at the candidate commit:

```bash
# 1. The full acceptance gate must be green.
CARGO_INCREMENTAL=0 make accept

# 2. Locally rebuild the reproducible Linux binaries and verify the SHA-256s
#    match. This is the same step CI runs.
REPRO_TARGET=x86_64-unknown-linux-musl \
  scripts/build_reproducible_release.sh
scripts/verify_reproducible_release.sh dist/linux-static-binaries

# 3. Regenerate the SBOM and confirm no new vulnerabilities slipped in.
python3 scripts/sbom.py --output dist/sbom.cdx.json
cargo audit --deny warnings \
  --ignore RUSTSEC-2024-0436 \
  --ignore RUSTSEC-2026-0009 \
  --ignore RUSTSEC-2025-0141

# 4. Run the docsite + rustdoc gates so the release publishes a fresh site.
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
make docsite

# 5. Confirm the cost ledger is within the $200 cap.
python3 python/cost_ledger.py check --path reports/cost.md --cap-usd 200

# 6. Confirm the parity tests pass against the reference dumps. Skip
#    gracefully without HF_TOKEN; CI runs the numerical pass with the cache.
HF_TOKEN="$HF_TOKEN" \
  cargo test -p lewm-core --features parity-fixtures -- parity_
```

Verify environment-level prerequisites:

- [ ] `git status` is clean and `git diff main..HEAD` is empty.
- [ ] `CHANGELOG.md` has a non-empty `[Unreleased]` section ready for
      promotion.
- [ ] `ROADMAP.md` does not flag any acceptance criterion as blocking the
      target version (see `## Definition of Full Completion`).
- [ ] `reports/release_checklist.md` items are all checked.
- [ ] `reports/cost.md` is up to date (any new HF Job has been appended via
      `python/cost_ledger.py append`).
- [ ] `HF_TOKEN` has been rotated within the last 90 days. If not, rotate
      before tagging.

---

## 3. Cutting the release

```bash
# 1. Promote [Unreleased] -> [vX.Y.Z] in CHANGELOG.md.
$EDITOR CHANGELOG.md

# 2. Bump workspace + per-crate versions.
$EDITOR Cargo.toml
cargo update --workspace --offline

# 3. Commit with conventional message + DCO.
git add -A
git commit -s -m "chore(release): prepare vX.Y.Z"

# 4. Tag and push. The release workflow triggers on the tag.
git tag -a "vX.Y.Z" -m "lewm-rs vX.Y.Z"
git push origin main vX.Y.Z
```

Once the tag is pushed, follow the workflow run at
`https://github.com/AbdelStark/lewm-rs/actions/workflows/release.yml`. The
pipeline has the following stages — each must finish green:

| Stage                 | Purpose                                                       |
|-----------------------|---------------------------------------------------------------|
| `tag-format`          | Validates the tag matches `^v[0-9]+\.[0-9]+\.[0-9]+$`.        |
| `build-linux-static`  | Reproducible static binary build under `ubuntu:22.04`.        |
| `build-macos-arm`     | aarch64-apple-darwin binary on `macos-14`.                    |
| `container`           | Builds + cosign-signs `ghcr.io/abdelstark/lewm-rs:vX.Y.Z`.     |
| `release-notes`       | Slices the latest `## [vX.Y.Z]` block from `CHANGELOG.md`.    |
| `sbom`                | CycloneDX SBOM via `scripts/sbom.py`.                         |
| `verify-reproducible` | Rebuilds the linux binary and bit-for-bit compares.           |
| `infer-export-verifier` | Runs the export-smoke test against the release binary.      |
| `attestation`         | GitHub built-in build provenance for binaries + SBOM.         |
| `github-release`      | Creates the **draft** release with all artifacts attached.    |
| `hub-models`          | Retags Hub artifacts when `HF_TOKEN` is configured.           |

When all stages finish:

1. Open the draft release. Review the auto-generated notes against
   `CHANGELOG.md` and edit as needed (link the relevant PRs, results
   reports, and HF artifact paths).
2. Confirm the attached artifacts:
   - `lewm-train`, `lewm-infer`, `lewm-plan` for linux and macOS.
   - `sbom.cdx.json`.
   - `release-notes.md`.
3. Verify the container image:
   ```bash
   cosign verify ghcr.io/abdelstark/lewm-rs:vX.Y.Z \
     --certificate-identity-regexp '^https://github.com/AbdelStark/lewm-rs' \
     --certificate-oidc-issuer https://token.actions.githubusercontent.com
   ```
4. Publish the draft.

---

## 4. Post-release tasks

- [ ] Open a `chore(release): start vX.Y.Z+1 cycle` PR that adds a fresh
      `[Unreleased]` heading to `CHANGELOG.md`.
- [ ] Update `ROADMAP.md` with the closed work and the next milestones.
- [ ] Refresh the Hugging Face Hub repo cards
      (`abdelstark/lewm-rs-pusht`, `abdelstark/lewm-rs-so100`,
      `abdelstark/lewm-rs-demo`) with the new release tag and link the
      release notes.
- [ ] Append the release cost row to `reports/cost.md` (release workflows
      run on GitHub-hosted minutes; the entry is informational at $0.00 but
      it keeps the audit trail unbroken).
- [ ] Announce on the project Discussion and tweet thread (template in
      `.github/release-templates/announce.md` if present).

---

## 5. Recovery / rollback

If the workflow fails after the container is pushed but before the draft is
published:

```bash
# Delete the container tag (cosign signature stays in the transparency log).
gh api -X DELETE "/users/abdelstark/packages/container/lewm-rs/versions/<ID>"

# Delete the tag locally and remotely.
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z

# Fix the underlying issue, push a new commit to main, retag.
```

Never edit a published release; cut a `+N` patch instead.

---

## 6. Audit trail

Every release leaves behind:

* The git tag (`vX.Y.Z`) with the signed commit.
* The draft-published GitHub release with binaries, SBOM, and release notes.
* The cosign signature on `ghcr.io/abdelstark/lewm-rs:vX.Y.Z` in the sigstore
  transparency log (Rekor).
* GitHub build attestations on each artifact (verifiable via
  `gh attestation verify <artifact> --owner AbdelStark`).
* The Hub artifact tree pointed at by the release notes.

Together these form the supply-chain receipt for the published version and
satisfy the SLSA L3-equivalent claims documented in RFC 0016.
