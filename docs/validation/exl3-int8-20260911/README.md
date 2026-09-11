# EXL3 INT8 experiment evidence

See the [validation report](../exl3-int8-20260911.md) for conclusions and the
[build instructions](../../../benchmarks/kernels/exl3_int8/README.md) for the
maintained reproduction entry point.

- `summary.json.gz` contains the final metrics and quality comparisons.
- `summarize.py.gz` verifies token inputs, pipeline layer indices, active kernel
  variants, library hashes, sample counts and cache misses.
- `prefill-model.json.gz` contains all request timings, generated outputs, model
  quality records and per-rank configuration. `prefill-model-inputs.json.gz`
  preserves the exact token IDs used by the historical FP16/NVFP4/AWQ comparison.
- `{dense,moe}-{vllm,exllamav3}-int8-{0,1,2}.json.gz` contains the independent
  decode-mode runs. Framework and extension versions are in `frameworks.json.gz`.
- `final-v9-*` contains the built source snapshots and metadata for the final
  tested library. `final-layer*.json.gz`, `final-sass-summary.json.gz` and
  `tests-sm80.log.gz` describe that build's kernel results, instructions and tests.
- `trace-*.log.gz` and the tracing source distinguish the active kernels. These
  traces were collected separately from performance measurements.
- `build-v*`, `final-build-*`, `v7-*` and `make_v*.py.gz` retain intermediate
  experiments, including rejected attempts. Intermediate generation scripts may
  precede later fixes; use the recorded built source snapshots for those results
  and the repository builder for the final implementation.
- `final-validation-status.json.gz` and `prefill-model-status.json.gz` record final
  command lines, environment overrides and exit statuses. Other status/log files
  belong to intermediate experiments.

The shared libraries are not checked in. Their SHA256 values are recorded in
build metadata. `manifest.json` hashes every evidence file except itself.
Gzip files preserve their original contents; decompress scripts and inputs into
an experiment directory before use. Historical scripts contain paths to this
machine's models, environments and `/tmp` artifacts.
