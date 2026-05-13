# Flamegraph Artifacts

CPU flamegraphs generated for performance work live under:

```text
profiling/flamegraphs/<git_sha>/<bench>.svg
```

Generate them with `scripts/run_local.sh flamegraph ...` so artifact names and
frame-pointer flags stay consistent with RFC 0014. Keep committed flamegraphs
small and targeted: one before/after pair for the hot path being optimized is
more useful than a directory full of exploratory captures.

The `sample/` directory contains a minimal reference SVG for reviewers and CI
artifact consumers. It is not a performance claim.
