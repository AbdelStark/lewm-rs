# Bench Baselines

Committed benchmark baselines live in `baselines.json`. They are generated from
Criterion output with:

```sh
cargo bench -p lewm-core --bench tensor_ops -- --save-baseline nightly
python3 scripts/bench_to_report.py --update-baseline --hardware self-hosted-l4
```

CI compares current `target/criterion/**/new/estimates.json` files against these
records for `TST-0014-BENCH-REG-001` and flags mean-time regressions greater
than five percent. A benchmark entry may carry `grace_started_at` in
`YYYY-MM-DD` format; regressions remain annotated but non-blocking for seven
days from that date. Pull request runs use the PR creation date when a baseline
entry does not carry an explicit grace date.
