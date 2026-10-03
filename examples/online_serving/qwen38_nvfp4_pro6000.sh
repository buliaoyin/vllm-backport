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
    "num_speculative_tokens": 3,
    "index_share_for_mtp_iteration": False,
    "mtp_token_map": sys.argv[1],
    "use_local_argmax_reduction": True,
    "enable_adaptive_verification": False,
}))
PY
)"

export TOKENIZERS_PARALLELISM=false

if [[ -n "${VLLM_EXE:-}" ]]; then
  VLLM_COMMAND=("$VLLM_EXE")
else
  VLLM_COMMAND=("$REPO_ROOT/.venv/bin/python" -m vllm.entrypoints.cli.main)
fi

exec "${VLLM_COMMAND[@]}" serve "$MODEL_PATH" \
  --served-model-name qwen38 --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.94 \
  --engram-config '{"cpu_offload":true}' \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-cache-dtype bfloat16 \
  --quantization-config '{"targets":{"*.linear_attn.in_proj_qkvz":"mxfp8","*.self_attn.qkv_proj":"mxfp8"}}' \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --hf-overrides '{"sm120_rowwise_fp8_head":true,"sm120_rowwise_fp8_hc":true,"sm120_rowwise_fp8_output":true}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,6,8,12,16]}' \
  "$@"
