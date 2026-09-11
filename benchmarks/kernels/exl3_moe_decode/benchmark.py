# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete expert decode wrappers with fixed routes and cold L2."""

import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.exl3_m32.benchmark import graph_times
from benchmarks.kernels.exl3_moe_decode.launcher import Decode
from vllm.model_executor.layers.quantization.exl3 import _exl3_moe_decode, _extension


def run(layer, method, args, k, n):
    sample = torch.load(args.route_sample, weights_only=True, map_location="cuda")
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "prefix": args.prefix,
        "dimensions": [k, n],
        "route_sample_sha256": hashlib.sha256(
            args.route_sample.read_bytes()
        ).hexdigest(),
        "library_sha256": hashlib.sha256(
            args.decode_kernel_library.read_bytes()
        ).hexdigest(),
        "timer": "15 CUDA graph replays, 256 MiB eviction before each timed interval",
        "route_note": "Small-M slices of the supplied capture; not a serving benchmark",
        "records": [],
    }
    for rows in args.decode_rows:
        if rows > sample["x"].shape[0]:
            raise ValueError("The route capture does not contain enough rows")
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
        reference = _exl3_moe_decode(*inputs)
        candidates = [("native_before", _exl3_moe_decode)]
        for mode in args.decode_modes:
            for grid in args.decode_grids:
                candidates.append(
                    (
                        f"{mode}_g{grid}",
                        Decode(
                            _extension(),
                            args.decode_kernel_library,
                            mode == "residual",
                            grid,
                        ),
                    )
                )
        candidates.append(("native_after", _exl3_moe_decode))
        for variant, implementation in candidates:
            call = partial(implementation, *inputs)

            actual = call()
            norm = reference.float().norm().item()
            error = (actual.float() - reference.float()).norm().item()
            relative = error / norm if norm else 0.0 if error == 0 else float("inf")
            record = {
                "rows": rows,
                "variant": variant,
                "relative_l2": relative,
                "finite": bool(torch.isfinite(actual).all()),
            }
            if record["finite"] and relative < 0.02:
                times = graph_times(call)
                record.update(times_ms=times, median_ms=statistics.median(times))
            else:
                record["rejected"] = "correctness"
            result["records"].append(record)
            Path(args.output).write_text(json.dumps(result, indent=2))
            print(
                {key: value for key, value in record.items() if key != "times_ms"},
                flush=True,
            )
