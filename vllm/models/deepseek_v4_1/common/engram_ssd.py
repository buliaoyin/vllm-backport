# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded direct-I/O Engram lookup from the original safetensors files."""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32

logger = init_logger(__name__)


def _capture_with_breaks():
    capture = BreakableCUDAGraphCapture.current()
    if capture is None or not capture._capturing:
        return None
    if is_forward_context_available() and (
        get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
    ):
        return None
    return capture


def checkpoint_slice(tensor: torch.Tensor) -> tuple[str, int]:
    """Locate a contiguous mmap view without touching its checkpoint pages."""
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Engram SSD requires contiguous CPU safetensors views")
    base = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        # Hugging Face snapshots point at suffixless blob files in /proc/maps.
        if (
            len(fields) != 6
            or fields[4] == "0"
            or not fields[1].endswith("p")
            or not fields[5].startswith("/")
        ):
            continue
        first, last = (int(x, 16) for x in fields[0].split("-"))
        if first <= base < base + size <= last:
            path = fields[5]
            if os.stat(path).st_ino != int(fields[4]):
                raise RuntimeError("Engram checkpoint changed during loading")
            return path, int(fields[2], 16) + base - first
    raise ValueError(
        "Engram SSD requires mmap safetensors loading; the source tensor "
        "is not a file-backed view"
    )


@lru_cache
def _library():
    from ..hybrid import native_libraries

    library = ctypes.CDLL(str(native_libraries()[1]))
    ptr, integer, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    if not hasattr(library, "dsv41_engram_ssd_version") or (
        library.dsv41_engram_ssd_version() != 1
    ):
        raise RuntimeError("Rebuild libdsv41_cuda to enable Engram SSD storage")
    signatures = {
        "create": (
            [ctypes.c_char_p, u64, ctypes.c_char_p, u64]
            + [integer] * 5
            + [u64]
            + [ptr] * 3,
            ptr,
        ),
        "error": ([ptr], ctypes.c_char_p),
        "task": ([ptr, integer], ptr),
        "enqueue": ([ptr, ptr], integer),
        "stats": ([ptr, ptr], None),
        "clear_cache": ([ptr], None),
        "destroy": ([ptr, ptr, integer], None),
    }
    for name, (args, result) in signatures.items():
        function = getattr(library, f"dsv41_engram_ssd_{name}")
        function.argtypes, function.restype = args, result
    return library


@triton.jit
def _unpack_rows(
    raw,
    status,
    output,
    rows,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # I/O failures must terminate inference, including during graph replay.
    tl.device_assert(tl.load(status) == 0, "Engram SSD I/O failed")
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    col = tl.arange(0, DIM)
    stride: tl.constexpr = DIM + DIM // QUANT_BLOCK
    values = tl.load(
        raw + row[:, None] * stride + col[None, :], mask=row[:, None] < rows, other=0
    )
    values = _decode_fp8_f32(values, False)
    scale = tl.load(
        raw + row[:, None] * stride + DIM + col[None, :] // QUANT_BLOCK,
        mask=row[:, None] < rows,
        other=0,
    )
    scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    tl.store(
        output + row[:, None] * DIM + col[None, :],
        (values * scale).to(tl.bfloat16),
        mask=row[:, None] < rows,
    )


class EngramSSD:
    """One serial I/O stream and fixed workspaces per local Engram table.

    Graph host nodes read the current IDs on every replay. Only row bytes are
    staged; request histories, speculation and rollback remain with the model.
    """

    def __init__(
        self, dim, block_size, heads, max_tokens, vocab_start, vocab_end, cache_bytes=0
    ):
        from ..host_memory import check_host_headroom

        self.dim, self.block_size, self.heads = dim, block_size, heads
        self.max_rows = max_tokens * heads
        self.start, self.end = vocab_start, vocab_end
        self.row_bytes = dim + dim // block_size
        self.cache_bytes = cache_bytes
        if self.max_rows * self.row_bytes > 2**32 - 1:
            raise ValueError("Engram SSD staging exceeds 4 GiB per table")
        check_host_headroom(
            self.max_rows * (self.row_bytes + 40) + 4 * 1024**2 + cache_bytes
        )
        self.library = _library()
        self.handle = None
        self.tasks = {}
        self.sources = {}
        self.stream = torch.cuda.Stream()
        self.ready = torch.cuda.Event()
        self.ids = torch.empty(
            self.max_rows, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.raw = torch.empty(
            self.max_rows,
            self.row_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self.status = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
        self.device_raw = torch.empty_like(self.raw, device="cuda")
        self.device_status = torch.empty_like(self.status, device="cuda")

    def load(self, param: torch.nn.Parameter, tensor: torch.Tensor):
        width = param.shape[1]
        expected_dtype = (
            torch.float8_e4m3fn if width == self.dim else torch.float8_e8m0fnu
        )
        if tensor.dtype not in (expected_dtype, torch.uint8):
            raise ValueError(f"Unexpected Engram SSD dtype: {tensor.dtype}")
        shard = tensor.narrow(0, self.start, self.end - self.start)
        if shard.shape != param.shape:
            raise ValueError("Engram SSD checkpoint shape does not match the model")
        self.sources[width] = checkpoint_slice(shard)
        if len(self.sources) != 2:
            return
        if self.handle is not None:
            raise RuntimeError("Engram SSD does not support in-place weight reload")
        weights, weight_offset = self.sources[self.dim]
        scales, scale_offset = self.sources[self.dim // self.block_size]
        self.handle = self.library.dsv41_engram_ssd_create(
            os.fsencode(weights),
            weight_offset,
            os.fsencode(scales),
            scale_offset,
            self.dim,
            self.dim // self.block_size,
            self.max_rows,
            self.start,
            self.end,
            self.cache_bytes,
            self.ids.data_ptr(),
            self.raw.data_ptr(),
            self.status.data_ptr(),
        )
        if not self.handle:
            raise RuntimeError(self.library.dsv41_engram_ssd_error(None).decode())
        logger.info(
            "Engram SSD: O_DIRECT + Linux AIO (not GDS), %.2f GiB on disk, "
            "host workspace ~%.1f MiB, GPU workspace %.1f MiB, "
            "bounded row cache %.1f MiB; %s",
            (self.end - self.start) * self.row_bytes / 1024**3,
            (self.max_rows * (self.row_bytes + 40) + 4 * 1024**2) / 1024**2,
            self.device_raw.numel() / 1024**2,
            self.cache_bytes / 1024**2,
            weights,
        )

    def check_errors(self):
        error = self.library.dsv41_engram_ssd_error(self.handle)
        if error:
            raise RuntimeError(error.decode())

    def lookup(self, indices, output, *, background=False):
        from vllm.utils.torch_utils import weak_ref_tensor

        capture = _capture_with_breaks()
        if capture is not None:
            # A side stream cannot remain forked when an attention graph segment
            # ends. Keep submission and consumption as explicit replay actions.
            ids, out = weak_ref_tensor(indices), weak_ref_tensor(output)
            capture.add_eager(lambda: self.lookup(ids, out, background=background))
            return
        if self.handle is None:
            raise RuntimeError("Engram SSD weights have not been loaded")
        self.check_errors()
        rows = indices.numel()
        if rows > self.max_rows or indices.dtype != torch.int32:
            raise ValueError("Engram SSD row IDs exceed workspace or are not int32")
        if not rows:
            return
        if rows not in self.tasks:
            self.tasks[rows] = self.library.dsv41_engram_ssd_task(self.handle, rows)
            if not self.tasks[rows]:
                raise RuntimeError("Cannot create Engram SSD graph host node")
        main = torch.cuda.current_stream()
        # Also fence the preceding forward's consumer before reusing buffers.
        # Captured paths fork and join this stream, including graph breaks.
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            self.ids[:rows].copy_(indices.reshape(-1), non_blocking=True)
            rc = self.library.dsv41_engram_ssd_enqueue(
                self.tasks[rows], self.stream.cuda_stream
            )
            if rc:
                raise RuntimeError(f"Cannot enqueue Engram SSD read: CUDA {rc}")
            self.device_raw[:rows].copy_(self.raw[:rows], non_blocking=True)
            self.device_status.copy_(self.status, non_blocking=True)
            _unpack_rows[(triton.cdiv(rows, 16),)](
                self.device_raw,
                self.device_status,
                output,
                rows,
                DIM=self.dim,
                QUANT_BLOCK=self.block_size,
                BLOCK_R=16,
                debug=True,
            )
            self.ready.record(self.stream)
        if not background:
            self.wait()

    def wait(self):
        capture = _capture_with_breaks()
        if capture is not None:
            capture.add_eager(self.wait)
            return
        torch.cuda.current_stream().wait_event(self.ready)

    def stats(self):
        """Return counters once all consuming streams have completed."""
        values = (ctypes.c_uint64 * 11)()
        self.library.dsv41_engram_ssd_stats(self.handle, values)
        return dict(
            zip(
                (
                    "calls",
                    "rows",
                    "valid_rows",
                    "weight_pages",
                    "scale_pages",
                    "read_bytes",
                    "read_ops",
                    "planning_ns",
                    "io_ns",
                    "callback_ns",
                    "cached_rows",
                ),
                values,
            )
        )

    def clear_cache(self):
        # Graph replay may execute the callback on an internal CUDA stream.
        torch.accelerator.synchronize(self.device_raw.device)
        self.library.dsv41_engram_ssd_clear_cache(self.handle)

    def close(self):
        if self.handle:
            tasks = (ctypes.c_void_p * len(self.tasks))(*self.tasks.values())
            self.library.dsv41_engram_ssd_destroy(self.handle, tasks, len(tasks))
            self.handle = None
            self.tasks.clear()

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()
