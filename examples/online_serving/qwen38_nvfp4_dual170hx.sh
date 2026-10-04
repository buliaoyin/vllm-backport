#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 /path/to/Qwen3.8-Flash-Next-NVFP4 [vllm arguments]}"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TOKEN_MAP="${MTP_TOKEN_MAP:-$REPO_ROOT/../sglang-rtxpro6000/configs/pennyroyal/frspec/flash-next-64k.pt}"
if [[ ! -f "$TOKEN_MAP" ]]; then
  printf 'Set MTP_TOKEN_MAP to the Flash-Next FR-Spec token map.\n' >&2
  exit 1
fi
SPECULATIVE_CONFIG="$("$REPO_ROOT/.venv/bin/python" - "$TOKEN_MAP" <<'PY'
import json
import sys

print(json.dumps({
    "method": "mtp",
    "num_speculative_tokens": 5,
    "index_share_for_mtp_iteration": False,
    "mtp_token_map": sys.argv[1],
    "use_local_argmax_reduction": True,
    "enable_adaptive_verification": True,
}))
PY
)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#GPU_IDS[@]}" -ne 2 ]]; then
  printf 'Set CUDA_VISIBLE_DEVICES to two CMP 170HX devices (e.g. 0,2).\n' >&2
  exit 1
fi
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ -n "${VLLM_EXE:-}" ]]; then
  VLLM_COMMAND=("$VLLM_EXE")
else
  VLLM_COMMAND=("$REPO_ROOT/.venv/bin/python" -m vllm.entrypoints.cli.main)
fi

exec "${VLLM_COMMAND[@]}" serve "$MODEL_PATH" \
  --served-model-name qwen38 --dtype bfloat16 \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.94 \
  --engram-config '{"cpu_offload":true}' \
  --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,8,9,10,12,15,16,18,20,24]}' \
  "$@"
