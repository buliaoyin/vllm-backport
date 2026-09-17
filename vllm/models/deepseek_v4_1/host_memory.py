# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, exact-size host allocations for resident hybrid weights."""

import ctypes
import math
import mmap
from contextlib import suppress
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4_numa import (
    host_memory_nodes,
)
from vllm.utils.numa_utils import get_libnuma

logger = init_logger(__name__)

GiB = 1024**3
LOAD_CHUNK_BYTES = 64 * 1024**2


def host_available_bytes():
    info = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return int(info["MemAvailable"].split()[0]) * 1024


def check_host_headroom(additional_bytes=0):
    if host_available_bytes() < 64 * GiB + additional_bytes:
        raise RuntimeError(
            "DeepSeek hybrid host allocation stopped before exhausting RAM: "
            "64 GiB of available memory must remain"
        )
    entry = next(
        (
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if entry is None:
        return
    path = Path("/sys/fs/cgroup") / entry.lstrip("/")
    while path != Path("/sys/fs"):
        limit_path = path / "memory.max"
        if limit_path.is_file():
            limit = limit_path.read_text().strip()
            used = int((path / "memory.current").read_text())
            if limit != "max" and used + additional_bytes + 8 * GiB > int(limit):
                raise RuntimeError(
                    "DeepSeek hybrid host allocation exceeds cgroup headroom"
                )
        path = path.parent


def interleave_host_mapping(owner):
    """Spread shared Engram/LRU pages before copying or CUDA registration."""
    nodes = host_memory_nodes()
    if len(nodes) < 2:
        return
    libnuma = get_libnuma()
    if libnuma is None:
        logger.warning_once("Host NUMA interleave unavailable: libnuma is missing")
        return
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = (ctypes.c_ulong * ((max(nodes) + bits) // bits))()
    for node in nodes:
        mask[node // bits] |= 1 << (node % bits)
    # MPOL_INTERLEAVE before first touch. Linux consumes maxnode - 1 mask bits.
    rc = libnuma.mbind(
        ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(owner))),
        ctypes.c_ulong(len(owner)),
        ctypes.c_int(3),
        mask,
        ctypes.c_ulong(len(mask) * bits + 1),
        ctypes.c_uint(0),
    )
    if rc:
        logger.warning_once(
            "Host NUMA interleave unavailable; Engram/LRU use inherited memory "
            "placement. Check container memory-policy permissions."
        )
    else:
        logger.info_once(
            "Hybrid Engram/LRU memory interleaved over NUMA nodes %s", nodes
        )


class RegisteredMapping(mmap.mmap):
    """Anonymous storage; registration grows only as checkpoint chunks arrive."""

    def __new__(cls, size):
        aligned = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        return super().__new__(
            cls, -1, aligned, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )

    def __init__(self, size):
        self.registered = []
        interleave_host_mapping(self)
        self.cudart = torch.cuda.cudart()

    def register(self, base, offset, size):
        if offset in self.registered:
            return
        size = min(
            len(self) - offset,
            (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE,
        )
        rc = self.cudart.cudaHostRegister(base + offset, size, 1)
        if int(rc):
            raise RuntimeError(
                f"Engram cudaHostRegister failed at offset {offset}: {rc}"
            )
        self.base = base
        self.registered.append(offset)

    def __del__(self):
        for offset in getattr(self, "registered", ()):
            with suppress(Exception):
                self.cudart.cudaHostUnregister(self.base + offset)


def empty_registered(shape, dtype, *, deferred=False):
    count = 1
    for n in shape:
        count *= n
    size = count * torch.empty((), dtype=dtype, device="cpu").element_size()
    owner = RegisteredMapping(size)
    tensor = torch.frombuffer(owner, dtype=dtype, count=count).reshape(shape)
    if not deferred:
        # A DMA copy may not straddle two host registrations. Expert LRU
        # transfers slice the first dimension, so keep each row in one span.
        alignment = math.lcm(size // shape[0], mmap.PAGESIZE)
        span = max(alignment, LOAD_CHUNK_BYTES // alignment * alignment)
        for offset in range(0, size, span):
            count = min(size - offset, span)
            check_host_headroom(count)
            owner.register(tensor.data_ptr(), offset, count)
    return tensor, owner
