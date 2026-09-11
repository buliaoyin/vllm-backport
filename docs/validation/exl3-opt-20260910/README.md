# EXL3 optimization artifacts

See [the report](../exl3-optimization-20260910.md) for results and interpretation.
`sha256.json` inventories every archived file; `provenance.json` records the
installed extension hash, upstream revision, current source hashes and hardware.
`validation.json` checks result completeness and identifies expected failures.

- `summary.json`: aggregated model timings, microbenchmark errors and checks.
- `moe-c512-sweep.json`, `moe-c1024-variants.json`: grouping, routing and linear
  cache ablations with the original layer placement.
- `equal-c1024.json`, `pp*-c*.json`: production code with alternative placement
  and scheduler/workspace capacities.
- `reuse-model.json`, `tile-model.json`: combined optimizations bracketed by
  repeated baselines. Baseline drift limits conclusions about small differences.
- `long-final.json`, `long-c2048.json`: 8K–64K production runs and the 64K larger
  chunk comparison. `long-final.json` also contains all 64 GSM8K completions.
- `eval-pp21.json`, `eval-equal.json`: repeated capacity/priority controls for
  the original four-question batch containing the differing GSM8K answer.
- `nvfp4-pp21-c1024.json`, `awq-pp18-c1024.json`, `exl3-pp18-c1024.json`:
  fresh comparisons with matching token IDs and matched placement within pairs.
- `*-status.json`: full commands, exit codes and available source snapshots.
  The AWQ 21-layer placement failed during allocation and has no latency result.
- `micro-*.json`, `linear-gpu*.json`, `hybrid-*.json`, `reuse-*-gpu*.json`,
  `tile-gpu*.json`: operator measurements; cases include repeated baselines.
- `baseline/`, `*-sources/`, `current-sources/`: source versions before and during
  the experiments. `production-optimization.patch.gz` isolates this turn's
  production/test edits from the already implemented integration.

Logs, Python drivers and CUDA experiments are gzip-compressed to preserve exact
text while keeping the archive small. The independent K16 sources derive from
ExLlamaV3 under `LICENSE-exllamav3.txt`; they are experimental and are not used by
the production backend. The private ABI launcher checks ExLlamaV3 1.4.8. Build
commands target SM80/SM120 with CUDA 13.0 and reuse pinned upstream headers.

The exact captured route tensors remain at:

- `/tmp/exl3-opt-20260910/routes/layer-3.pt`
- `/tmp/exl3-opt-20260910/routes/layer-22.pt`
- `/tmp/exl3-opt-20260910/routes/layer-44.pt`

Their hashes and sizes are in `provenance.json`. They and compiled `.so` files
are not duplicated into the repository. To regenerate routing samples, run the
benchmark with the same 8K input, original PP=11/11/11/12, outer chunk=512,
`VLLM_EXL3_MOE_PRIORITY=0`, `--moe-workspace-sizes 128 256 512` and
`--capture-routing-dir /tmp/exl3-opt-20260910/routes`. Use the archived capacity
sweep command for all other flags. Atomic accumulation may produce minor
floating-point differences between new captures; the recorded local tensors
identify the exact samples used for these results.

To replay the archived temporary drivers on this machine, decompress their
`.py.gz`, `.sh.gz` and CUDA sources into `/tmp/exl3-opt-20260910`, maintaining
relative names, and use `.venv/bin/python` from the repository. Kernel drivers
require the supplied checkpoint, route tensors and matching installed extension.
Run GPU jobs serially. A new hardware/software setup needs new measurements.
