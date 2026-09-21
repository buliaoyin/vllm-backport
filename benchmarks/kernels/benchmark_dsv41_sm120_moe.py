# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Marlin and installed SM120 CUTLASS MoE on V4.1 expert weights."""

import argparse
import json
import os
import statistics
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from cpu.benchmark_dsv41_cpu_moe import reference
from flashinfer import mxfp8_quantize
from flashinfer.testing import bench_gpu_time_with_cudagraph
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe


def native_mxfp8_moe(input, **kwargs):
    quantized, scales = mxfp8_quantize(
        input, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"
    )
    return flashinfer_cutlass_fused_moe(input=quantized, input_sf=scales, **kwargs)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--experts", type=int, default=240)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 128])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.accelerator.set_device_index(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", args.device)
    assert torch.cuda.get_device_capability(device) == (12, 0)
    payload = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu": torch.cuda.get_device_name(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
        "timing": "CUDA graph events; complete routed MoE; cold L2 rotating buffers",
        "results": [],
    }

    def save():
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    weights, handles = {}, {}
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    with ExitStack() as stack:
        for expert in range(args.experts):
            for projection in range(3):
                pair = []
                for suffix in ("weight", "scale"):
                    key = (
                        f"layers.{args.layer}.ffn.experts.{expert}."
                        f"w{projection + 1}.{suffix}"
                    )
                    filename = index["weight_map"][key]
                    if filename not in handles:
                        handles[filename] = stack.enter_context(
                            safe_open(args.model / filename, framework="pt")
                        )
                    pair.append(handles[filename].get_tensor(key).view(torch.uint8))
                weights[expert, projection] = tuple(pair)
        raw = tuple(
            torch.stack(
                [
                    torch.cat([weights[e, p][part] for p in projections], dim=0)
                    for e in range(args.experts)
                ]
            ).to(device)
            for projections, part in (((0, 2), 0), ((1,), 0), ((0, 2), 1), ((1,), 1))
        )
        layer = SimpleNamespace(params_dtype=torch.bfloat16)
        prepared = {
            "marlin": prepare_moe_mxfp4_layer_for_marlin(
                layer,
                *(t.clone() for t in raw),
                None,
                None,
                inplace=True,
            )[:4]
        }
        conversion_errors = {
            "cutlass_bf16": "Installed vLLM supports MXFP4/BF16 only on SM90"
        }
        for name, backend in [
            ("cutlass_mxfp8", Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8),
        ]:
            try:
                prepared[name] = convert_weight_to_mxfp4_moe_kernel_format(
                    backend, layer, *raw
                )[:4]
            except Exception as error:
                conversion_errors[name] = repr(error)
        payload["conversion_errors"] = conversion_errors
        save()
        marlin_workspace = marlin_make_workspace_new(device, 4)
        native_workspace = torch.empty(512 * 1024**2, dtype=torch.uint8, device=device)
        one = torch.ones(args.experts, device=device)
        fake_input_scale = torch.ones(args.experts, device=device)
        zero = torch.zeros_like(one)
        limit = torch.full_like(one, 10.0)
        for tokens in args.tokens:
            hidden_cpu = torch.randn(tokens, 5120, dtype=torch.bfloat16)
            ids_cpu = torch.stack(
                [torch.randperm(args.experts)[:6] for _ in range(tokens)]
            ).int()
            routes_cpu = torch.rand(tokens, 6).softmax(-1) * 2.5
            expected = reference(hidden_cpu, ids_cpu, routes_cpu, weights, 10.0, device)
            hidden, ids, routes = (
                t.to(device) for t in (hidden_cpu, ids_cpu, routes_cpu)
            )
            operations, errors = {}, {}
            for name, (w13, w2, s13, s2) in prepared.items():
                if name == "marlin":
                    operation = partial(
                        fused_marlin_moe,
                        hidden,
                        w13,
                        w2,
                        None,
                        None,
                        s13,
                        s2,
                        routes,
                        ids,
                        scalar_types.float4_e2m1f.id,
                        global_num_experts=args.experts,
                        workspace=marlin_workspace,
                        activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
                    )
                else:
                    mxfp8 = name == "cutlass_mxfp8"
                    operation = partial(
                        native_mxfp8_moe,
                        input=hidden,
                        token_selected_experts=ids,
                        token_final_scales=routes,
                        fc1_expert_weights=w13.view(torch.long) if mxfp8 else w13,
                        fc2_expert_weights=w2.view(torch.long) if mxfp8 else w2,
                        output=torch.empty_like(hidden),
                        output_dtype=torch.bfloat16,
                        quant_scales=[
                            s13.view(torch.int32),
                            fake_input_scale,
                            s2.view(torch.int32),
                            fake_input_scale,
                        ]
                        if mxfp8
                        else [s13, s2],
                        swiglu_alpha=one,
                        swiglu_beta=zero,
                        swiglu_limit=limit,
                        use_mxfp8_act_scaling=mxfp8,
                        use_w4_group_scaling=not mxfp8,
                        use_fused_finalize=False,
                        workspace_buffer=native_workspace,
                    )
                try:
                    result = operation()
                    actual = result[0] if isinstance(result, list) else result
                    actual = actual.float().cpu()
                    rms = ((actual - expected).norm() / expected.norm()).item()
                    assert torch.isfinite(actual).all() and rms < 0.035, rms
                    errors[name] = {"relative_rms": rms}
                    operations[name] = operation
                except Exception as error:
                    if name == "marlin":
                        raise
                    errors[name] = {"error": repr(error)}
            record = {"tokens": tokens, "correctness": errors, "median_ms_by_order": {}}
            payload["results"].append(record)
            save()
            print(json.dumps(record), flush=True)
            for order in (list(operations), list(reversed(operations))):
                for name in order:
                    op = operations[name]
                    try:
                        samples = bench_gpu_time_with_cudagraph(
                            op.func,
                            input_args=op.args,
                            input_kwargs=op.keywords,
                            cold_l2_cache=True,
                            num_iters_within_graph=4,
                            dry_run_iters=3,
                            repeat_iters=15,
                        )
                        record["median_ms_by_order"].setdefault(name, []).append(
                            statistics.median(samples)
                        )
                    except Exception as error:
                        record.setdefault("timing_errors", {})[name] = repr(error)
                    save()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
