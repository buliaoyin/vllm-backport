# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare EXL3 fused experts and small-batch GEMM on a checkpoint MoE layer."""

import argparse
import itertools
import json
import statistics
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from flashinfer.testing import bench_gpu_time_with_cudagraph
from safetensors import safe_open

from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3MoEMethod,
    _exl3_moe_decode,
    _exl3_moe_fused,
    _extension,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _route_counts(
    Ids, Counts, Assignments: tl.constexpr, Experts: tl.constexpr, Block: tl.constexpr
):
    expert = tl.program_id(0)
    offsets = tl.arange(0, Block)
    ids = tl.load(Ids + offsets, offsets < Assignments, -1)
    count = tl.sum((ids == expert).to(tl.int32), 0)
    tl.store(Counts + expert, count)


@triton.jit
def _route_pack(
    Ids,
    Weights,
    Counts,
    Tokens,
    SortedWeights,
    Pointers,
    SortedPointers,
    Assignments: tl.constexpr,
    Experts: tl.constexpr,
    TopK: tl.constexpr,
    Block: tl.constexpr,
    ExpertBlock: tl.constexpr,
    Longest: tl.constexpr,
):
    expert = tl.program_id(0)
    expert_offsets = tl.arange(0, ExpertBlock)
    counts = tl.load(Counts + expert_offsets, expert_offsets < Experts, 0)
    count = tl.load(Counts + expert)
    if Longest:
        precedes = (counts > count) | ((counts == count) & (expert_offsets < expert))
        precedes = precedes & (expert_offsets < Experts)
        rank = tl.sum(precedes.to(tl.int32), 0)
        offset = tl.sum(tl.where(precedes, counts, 0), 0)
        for component in range(9):
            ptr = tl.load(Pointers + component * Experts + expert)
            tl.store(SortedPointers + component * Experts + rank, ptr)
        tl.store(SortedPointers + 9 * Experts + rank, count)
    else:
        offset = tl.sum(tl.where(expert_offsets < expert, counts, 0), 0)
    offsets = tl.arange(0, Block)
    ids = tl.load(Ids + offsets, offsets < Assignments, -1)
    selected = ids == expert
    positions = offset + tl.cumsum(selected.to(tl.int32), 0) - 1
    weights = tl.load(Weights + offsets, offsets < Assignments, 0)
    tl.store(Tokens + positions, offsets // TopK, selected)
    tl.store(SortedWeights + positions, weights, selected)


def triton_routes(ids, weights, ptrs, topk, longest_first):
    experts = ptrs[0].numel()
    counts = torch.empty(experts + 1, dtype=torch.int64, device=ids.device)
    tokens = torch.empty_like(ids)
    sorted_weights = torch.empty_like(weights)
    pointers = torch.stack(ptrs) if longest_first else ptrs[0]
    # Last row is the permuted histogram; keep a zero sentinel for the extension.
    # Packed pointer stride excludes the trailing histogram sentinel.
    packed = (
        torch.zeros(10 * experts + 1, dtype=torch.int64, device=ids.device)
        if longest_first
        else pointers
    )
    _route_counts[(experts + 1,)](
        ids, counts, ids.numel(), experts, triton.next_power_of_2(ids.numel())
    )
    _route_pack[(experts,)](
        ids,
        weights,
        counts,
        tokens,
        sorted_weights,
        pointers,
        packed,
        ids.numel(),
        experts,
        topk,
        triton.next_power_of_2(ids.numel()),
        triton.next_power_of_2(experts),
        longest_first,
    )
    if longest_first:
        ptrs = list(packed[: 9 * experts].view(9, experts).unbind(0))
        counts = packed[9 * experts :]
    return counts, tokens, sorted_weights, ptrs


def fused_with_chunk(
    inputs,
    chunk_size,
    groups=-1,
    splits=1,
    split_threshold=0,
    packing="sort",
    longest_first=False,
):
    """Ablate token grouping only; callers must check workspace capacity."""
    x, topk_weights, topk_ids, ptrs, workspace, bits, flags, limit = inputs
    ext = _extension()
    output = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk_size):
        hidden = x[start : start + chunk_size].to(torch.float16).contiguous()
        ids = topk_ids[start : start + chunk_size].to(torch.int64).flatten()
        weights = topk_weights[start : start + chunk_size].to(torch.float16).flatten()
        chunk_ptrs = ptrs
        if splits > 1:
            sub_ids = (
                torch.arange(hidden.shape[0], device=x.device).repeat_interleave(
                    topk_ids.shape[1]
                )
                % splits
            )
            if split_threshold:
                original_counts = torch.zeros(
                    ptrs[0].numel() // splits + 1, dtype=torch.int64, device=x.device
                )
                original_counts.scatter_add_(0, ids, torch.ones_like(ids))
                sub_ids = torch.where(
                    original_counts[ids] >= split_threshold, sub_ids, 0
                )
            ids = ids * splits + sub_ids
        if packing == "triton":
            counts, tokens, sorted_weights, chunk_ptrs = triton_routes(
                ids, weights, chunk_ptrs, topk_ids.shape[1], longest_first
            )
        else:
            counts = torch.zeros(
                ptrs[0].numel() + 1, dtype=torch.int64, device=x.device
            )
            counts.scatter_add_(0, ids, torch.ones_like(ids))
            if longest_first:
                permutation = torch.argsort(counts[:-1], descending=True, stable=True)
                inverse = torch.empty_like(permutation)
                inverse.scatter_(
                    0, permutation, torch.arange(permutation.numel(), device=x.device)
                )
                ids = inverse[ids]
                counts = torch.cat((counts[:-1][permutation], counts[-1:]))
                chunk_ptrs = [ptr[permutation] for ptr in ptrs]
            if packing == "align":
                from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
                    moe_align_block_size,
                )

                order = moe_align_block_size(
                    ids.view(-1, topk_ids.shape[1]), 1, ptrs[0].numel()
                )[0].long()
            else:
                order = torch.argsort(ids, stable=True)
            tokens = torch.div(order, topk_ids.shape[1], rounding_mode="floor")
            sorted_weights = weights[order]
        result = torch.zeros_like(hidden, dtype=torch.float32)
        ext.exl3_moe(
            hidden,
            result,
            counts,
            tokens,
            sorted_weights,
            *workspace,
            0,
            *bits,
            *chunk_ptrs,
            *flags,
            limit,
            groups,
        )
        output[start : start + hidden.shape[0]].copy_(result)
    return output


def fused_with_reused_buffers(
    inputs,
    chunk_size,
    groups=-1,
    splits=1,
    split_threshold=0,
    packing="sort",
    longest_first=True,
    *,
    pool,
):
    """Reuse bounded internal buffers; every returned output still owns its storage."""
    assert splits == 1 and not split_threshold and packing == "sort"
    x, topk_weights, topk_ids, ptrs, workspace, bits, flags, limit = inputs
    ext = _extension()
    experts, topk = ptrs[0].numel(), topk_ids.shape[1]
    key = (x.device, chunk_size, x.shape[1], experts, topk)
    if key not in pool:

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=x.device)

        pool[key] = {
            "hidden": empty((chunk_size, x.shape[1]), torch.float16),
            "result": empty((chunk_size, x.shape[1]), torch.float32),
            "ids": empty(chunk_size * topk, torch.int64),
            "mapped_ids": empty(chunk_size * topk, torch.int64),
            "weights": empty(chunk_size * topk, torch.float16),
            "sorted_weights": empty(chunk_size * topk, torch.float16),
            "counts": empty(experts + 1, torch.int64),
            "sorted_counts": empty(experts + 1, torch.int64),
            "permutation": empty(experts, torch.int64),
            "inverse": empty(experts, torch.int64),
            "sorted_ids": empty(chunk_size * topk, torch.int64),
            "order": empty(chunk_size * topk, torch.int64),
            "tokens": empty(chunk_size * topk, torch.int64),
            "ones": torch.ones(chunk_size * topk, dtype=torch.int64, device=x.device),
            "range": torch.arange(experts, device=x.device),
            "ptrs": [torch.empty_like(p) for p in ptrs],
        }
    scratch = pool[key]
    output = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk_size):
        rows = min(chunk_size, x.shape[0] - start)
        slots = rows * topk
        hidden = scratch["hidden"][:rows]
        hidden.copy_(x[start : start + rows])
        ids = scratch["ids"][:slots]
        ids.copy_(topk_ids[start : start + rows].flatten())
        weights = scratch["weights"][:slots]
        weights.copy_(topk_weights[start : start + rows].flatten())
        counts = scratch["counts"].zero_()
        counts.scatter_add_(0, ids, scratch["ones"][:slots])
        chunk_ptrs = ptrs
        if longest_first:
            sorted_counts = scratch["sorted_counts"]
            sorted_counts[-1:].zero_()
            permutation = scratch["permutation"]
            torch.sort(
                counts[:-1],
                stable=True,
                descending=True,
                out=(sorted_counts[:-1], permutation),
            )
            inverse = scratch["inverse"]
            inverse.scatter_(0, permutation, scratch["range"])
            torch.index_select(inverse, 0, ids, out=scratch["mapped_ids"][:slots])
            ids = scratch["mapped_ids"][:slots]
            counts = sorted_counts
            chunk_ptrs = scratch["ptrs"]
            for source, dest in zip(ptrs, chunk_ptrs):
                torch.index_select(source, 0, permutation, out=dest)
        order = scratch["order"][:slots]
        torch.sort(ids, stable=True, out=(scratch["sorted_ids"][:slots], order))
        tokens = scratch["tokens"][:slots]
        torch.div(order, topk, rounding_mode="floor", out=tokens)
        sorted_weights = scratch["sorted_weights"][:slots]
        torch.index_select(weights, 0, order, out=sorted_weights)
        result = scratch["result"][:rows].zero_()
        ext.exl3_moe(
            hidden,
            result,
            counts,
            tokens,
            sorted_weights,
            *workspace,
            0,
            *bits,
            *chunk_ptrs,
            *flags,
            limit,
            groups,
        )
        output[start : start + rows].copy_(result)
    return output


@triton.jit
def _hot_grouped_gemm(
    A,
    B,
    C,
    Counts,
    Experts,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(1)
    tile = tl.program_id(0)
    row_tile = tile // tl.cdiv(N, BN)
    col_tile = tile % tl.cdiv(N, BN)
    expert = tl.load(Experts + group)
    count = tl.load(Counts + expert)
    if row_tile * BM < count:
        rows = row_tile * BM + tl.arange(0, BM)
        cols = col_tile * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            offsets_k = start * BK + ks
            a = tl.load(
                A + group * M * K + rows[:, None] * K + offsets_k[None, :],
                (rows[:, None] < count) & (offsets_k[None, :] < K),
                0,
            )
            b = tl.load(
                B + group * K * N + offsets_k[:, None] * N + cols[None, :],
                (offsets_k[:, None] < K) & (cols[None, :] < N),
                0,
            )
            accumulator += tl.dot(a, b)
        tl.store(
            C + group * M * N + rows[:, None] * N + cols[None, :],
            accumulator.to(C.dtype.element_ty),
            (rows[:, None] < count) & (cols[None, :] < N),
        )


def hybrid_with_chunk(inputs, chunk_size, hot_count, projections, grouped=False):
    """Reconstruct a fixed number of GPU-selected hot experts, with no CPU readback."""
    x, weights, ids, ptrs, workspace, bits, flags, limit = inputs
    ext = _extension()
    output = torch.empty_like(x)
    experts = ptrs[0].numel()
    for start in range(0, x.shape[0], chunk_size):
        hidden = x[start : start + chunk_size]
        chunk_ids = ids[start : start + chunk_size].long()
        chunk_weights = weights[start : start + chunk_size]
        counts = torch.zeros(experts + 1, dtype=torch.int64, device=x.device)
        counts.scatter_add_(
            0, chunk_ids.flatten(), torch.ones_like(chunk_ids.flatten())
        )
        hot = counts[:-1].topk(hot_count).indices
        selected = torch.zeros(experts + 1, dtype=torch.bool, device=x.device)
        selected.index_fill_(0, hot, True)
        cold_ids = torch.where(selected[chunk_ids], experts, chunk_ids)
        cold_inputs = (
            hidden.float(),
            chunk_weights,
            cold_ids,
            ptrs,
            workspace,
            bits,
            flags,
            limit,
        )
        result = fused_with_chunk(cold_inputs, chunk_size)
        order = torch.argsort(chunk_ids.flatten(), stable=True)
        offsets = counts.cumsum(0) - counts
        positions = (
            offsets[hot, None] + torch.arange(hidden.shape[0], device=x.device)[None, :]
        )
        valid = (
            torch.arange(hidden.shape[0], device=x.device)[None, :] < counts[hot, None]
        )
        assignment = order[positions.clamp(max=order.numel() - 1)]
        tokens = assignment // chunk_ids.shape[1]
        routing = chunk_weights.flatten()[assignment].half() * valid
        gathered = hidden[tokens].half()

        def project(value, index, hot=hot, hidden=hidden, counts=counts):
            packed, suh, svh = (t[hot].contiguous() for t in projections[index])
            k, n = suh.shape[1], svh.shape[1]
            matrix = torch.empty(
                (hot_count, k, n), dtype=torch.float16, device=x.device
            )
            for i in range(hot_count):
                ext.reconstruct_slice(
                    matrix[i],
                    packed[i],
                    bits[index],
                    *flags[index * 2 : index * 2 + 2],
                    0,
                )
            rotated = torch.empty_like(value)
            ext.had_r_128(
                (value * suh[:, None, :]).reshape(-1, k),
                rotated.view(-1, k),
                None,
                None,
                1.0,
            )
            if grouped:
                projected = torch.empty(
                    (hot_count, hidden.shape[0], n),
                    device=x.device,
                    dtype=torch.float16,
                )
                _hot_grouped_gemm[
                    (triton.cdiv(hidden.shape[0], 32) * triton.cdiv(n, 128), hot_count)
                ](
                    rotated,
                    matrix,
                    projected,
                    counts,
                    hot,
                    hidden.shape[0],
                    n,
                    k,
                    32,
                    128,
                    64,
                    num_warps=4,
                )
            else:
                projected = torch.bmm(rotated, matrix)
            ext.had_r_128(projected.view(-1, n), projected.view(-1, n), None, None, 1.0)
            return projected * svh[:, None, :]

        gate = project(gathered, 0)
        up = project(gathered, 1)
        if limit:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        activated = (torch.nn.functional.silu(gate.float()) * up).half()
        down = project(activated, 2).masked_fill(~valid[:, :, None], 0)
        result.index_add_(
            0,
            tokens.flatten(),
            (down.float() * routing[:, :, None]).reshape(-1, x.shape[1]),
        )
        output[start : start + hidden.shape[0]].copy_(result)
    return output


def hybrid_ablation(layer, method, args, k, n):
    projections = []
    for kind in ("w1", "w3", "w2"):
        part = []
        for component in ("trellis", "suh", "svh"):
            name, offset, size, shape = method.views[kind, component]
            part.append(
                getattr(layer, name)
                .data.narrow(0, offset, size * args.experts)
                .view(args.experts, *shape)
            )
        projections.append(part)
    sample = torch.load(args.route_sample, weights_only=True, map_location="cuda")
    result = {
        "gpu": torch.cuda.get_device_name(),
        "routing": str(args.route_sample),
        "timer": "CUDA graph events",
        "hot_selection": "GPU top-k per chunk, no host sync",
        "rows": [],
    }
    for rows in args.prefill_rows:
        x, weights, ids = (
            sample[name][:rows].contiguous() for name in ("x", "weights", "ids")
        )
        base_inputs = (
            x,
            weights,
            ids,
            method.ptrs,
            method.workspace,
            method.bits,
            method.flags,
            10.0,
        )
        reference = fused_with_chunk(base_inputs, 128)
        for chunk, hot_count in itertools.product(
            args.prefill_chunks, args.hybrid_experts
        ):
            workspace = [
                t.new_empty((t.shape[0], chunk, t.shape[2])) for t in method.workspace
            ]
            inputs = (*base_inputs[:4], workspace, *base_inputs[5:])

            def calculate(inputs, chunk=chunk, hot_count=hot_count):
                return (
                    hybrid_with_chunk(
                        inputs, chunk, hot_count, projections, args.hybrid_grouped
                    )
                    if hot_count
                    else fused_with_chunk(inputs, chunk)
                )

            actual = calculate(inputs)
            relative = (
                (actual.float() - reference.float()).norm() / reference.float().norm()
            ).item()
            assert relative < 0.01, (rows, chunk, hot_count, relative)
            for _ in range(3):
                calculate(inputs)
            times = bench_gpu_time_with_cudagraph(
                calculate,
                input_args=(inputs,),
                cold_l2_cache=True,
                num_iters_within_graph=1,
                dry_run_iters=3,
                repeat_iters=15,
            )
            record = {
                "tokens": rows,
                "chunk": chunk,
                "hot_experts": hot_count,
                "grouped": args.hybrid_grouped,
                "relative_l2": relative,
                "median_ms": statistics.median(times),
            }
            result["rows"].append(record)
            print(record, flush=True)
            args.output.write_text(json.dumps(result, indent=2))


def prefill_ablation(layer, method, args, k, n):
    candidate_fn = (
        partial(fused_with_reused_buffers, pool={})
        if args.reuse_buffers
        else fused_with_chunk
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "prefix": args.prefix,
        "checkpoint": str(args.checkpoint),
        "expand_workspace": args.expand_workspace,
        "shape": {"hidden": k, "intermediate": n, "experts": args.experts, "topk": 8},
        "seed": 20260910,
        "routing": str(args.route_sample) if args.route_sample else args.routing,
        "reuse_buffers": args.reuse_buffers,
        "cold_l2": True,
        "timer": "CUDA graph events (CUPTI unavailable on CMP)",
        "scope": "full expert wrapper GPU work, excludes Python replay dispatch",
        "graph_calls_per_sample": 1,
        "samples": 15,
        "flops": "6 * tokens * topk * hidden * intermediate (useful GEMM only)",
        "rows": [],
    }
    torch.manual_seed(result["seed"])
    for rows in args.prefill_rows:
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        scores = torch.rand(rows, args.experts, device="cuda")
        if args.routing == "hot":
            scores[:, 0] = 2.0
        ids = scores.topk(8, dim=-1).indices
        weights = torch.rand(rows, 8, device="cuda").softmax(-1)
        if args.route_sample:
            sample = torch.load(
                args.route_sample, weights_only=True, map_location="cuda"
            )
            assert rows <= sample["x"].shape[0]
            x, weights, ids = (
                sample[name][:rows].contiguous() for name in ("x", "weights", "ids")
            )
        inputs = (
            x,
            weights,
            ids,
            method.ptrs,
            method.workspace,
            method.bits,
            method.flags,
            10.0,
        )
        reference = fused_with_chunk(inputs, 128)
        record = {"tokens": rows, "chunks": []}
        candidates = list(
            itertools.product(
                args.prefill_chunks,
                args.prefill_groups,
                args.expert_splits,
                args.split_thresholds,
                args.packing,
                args.longest_first,
            )
        )
        for chunk, groups, splits, threshold, packing, longest_first in candidates:
            if splits == 1 and threshold:
                continue
            counts = torch.stack(
                [
                    torch.bincount(part.flatten(), minlength=args.experts)
                    for part in ids.split(chunk)
                ]
            )
            peak = counts.max().item()
            capacity = min(chunk, rows) if args.expand_workspace else 128
            workspace = method.workspace
            if capacity != workspace[0].shape[1]:
                workspace = [
                    t.new_empty((t.shape[0], capacity, t.shape[2])) for t in workspace
                ]
            pointers = (
                [ptr.repeat_interleave(splits) for ptr in method.ptrs]
                if splits > 1
                else method.ptrs
            )
            candidate_inputs = (*inputs[:3], pointers, workspace, *inputs[5:])
            options = (chunk, groups, splits, threshold, packing, bool(longest_first))
            item = {
                "groups": groups,
                "splits": splits,
                "split_threshold": threshold,
                "packing": packing,
                "longest_first": bool(longest_first),
                "workspace_rows": capacity,
                "workspace_bytes": sum(t.numel() * t.element_size() for t in workspace),
                "chunk": chunk,
                "max_tokens_per_expert": peak,
                "active_expert_visits": (counts > 0).sum().item(),
                "rows_in_16row_tiles": (((counts + 15) // 16) * 16).sum().item(),
            }
            if peak > capacity:
                item["skipped"] = "would silently drop experts exceeding workspace rows"
            else:
                candidate = candidate_fn(candidate_inputs, *options)
                relative = (
                    (candidate.float() - reference.float()).norm()
                    / reference.float().norm()
                ).item()
                assert relative < 0.01, (rows, chunk, relative)
                item["relative_l2"] = relative
                for _ in range(3):
                    candidate_fn(candidate_inputs, *options)
                torch.accelerator.synchronize()
                ms = bench_gpu_time_with_cudagraph(
                    candidate_fn,
                    input_args=(candidate_inputs, *options),
                    cold_l2_cache=True,
                    num_iters_within_graph=1,
                    dry_run_iters=3,
                    repeat_iters=15,
                )
                item["median_ms"] = statistics.median(ms)
                item["min_ms"] = min(ms)
                item["max_ms"] = max(ms)
                item["useful_tflops"] = 6 * rows * 8 * k * n / (item["median_ms"] * 1e9)
            record["chunks"].append(item)
            print(rows, item, flush=True)
        result["rows"].append(record)
        args.output.write_text(json.dumps(result, indent=2))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-rows", type=int, nargs="+")
    parser.add_argument("--expand-workspace", action="store_true")
    parser.add_argument("--routing", choices=["uniform", "hot"], default="uniform")
    parser.add_argument("--route-sample", type=Path)
    parser.add_argument("--hybrid-experts", type=int, nargs="+")
    parser.add_argument("--hybrid-grouped", action="store_true")
    parser.add_argument("--reuse-buffers", action="store_true")
    parser.add_argument("--prefill-groups", type=int, nargs="+", default=[-1])
    parser.add_argument("--expert-splits", type=int, nargs="+", default=[1])
    parser.add_argument("--split-thresholds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--packing", choices=["sort", "align", "triton"], nargs="+", default=["sort"]
    )
    parser.add_argument(
        "--longest-first", type=int, choices=[0, 1], nargs="+", default=[0]
    )
    parser.add_argument(
        "--prefill-chunks", type=int, nargs="+", default=[128, 512, 2048]
    )
    args = parser.parse_args()
    config = Exl3Config({})
    config.maybe_update_config(str(args.checkpoint))
    k, n = config.matrices[args.prefix + ".0.gate_proj"].dimensions
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1),
        activation="silu",
        swiglu_limit=10.0,
    )
    layer = torch.nn.Module()
    method = Exl3MoEMethod(config, moe, args.prefix)
    with torch.device("cuda"):
        method.create_weights(layer, args.experts, k, n, torch.bfloat16)
    for shard in args.checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in list(handle.keys()):
                if not name.startswith(args.prefix + "."):
                    continue
                expert, proj, component = name[len(args.prefix) + 1 :].split(".")
                if int(expert) >= args.experts:
                    continue
                kind = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[proj]
                param = getattr(layer, ("w2_" if kind == "w2" else "w13_") + component)
                param.weight_loader(
                    param, handle.get_tensor(name), expert_id=int(expert), shard_id=kind
                )
    method.process_weights_after_loading(layer)
    if args.hybrid_experts:
        hybrid_ablation(layer, method, args, k, n)
        return
    if args.prefill_rows:
        prefill_ablation(layer, method, args, k, n)
        return
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "prefix": args.prefix,
        "cold_l2": True,
        "timer": "CUDA graph",
        "rows": [],
    }
    for rows in [1, 4, 8]:
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(rows, args.experts, device="cuda").topk(8, dim=-1).indices
        weights = torch.rand(rows, 8, device="cuda").softmax(-1)
        inputs = (
            x,
            weights,
            ids,
            method.ptrs,
            method.workspace,
            method.bits,
            method.flags,
            10.0,
        )
        fused = _exl3_moe_fused(*inputs)
        decode = _exl3_moe_decode(*inputs)
        rel = ((fused.float() - decode.float()).norm() / fused.float().norm()).item()
        assert rel < 0.01, rel
        record = {"tokens": rows, "relative_l2": rel}
        for label, fn in [
            ("fused_us", _exl3_moe_fused),
            ("decode_us", _exl3_moe_decode),
        ]:
            for _ in range(10):
                fn(*inputs)
            torch.accelerator.synchronize()
            timings = bench_gpu_time_with_cudagraph(
                fn,
                input_args=inputs,
                cold_l2_cache=True,
                dry_run_iters=5,
                repeat_iters=40,
            )
            record[label] = statistics.median(timings) * 1000
        result["rows"].append(record)
        print(record, flush=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
