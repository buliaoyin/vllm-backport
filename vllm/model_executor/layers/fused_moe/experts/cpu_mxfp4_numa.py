# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU topology for resident MXFP4 experts; respect the caller's CPU set."""

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger
from vllm.utils.numa_utils import get_libnuma

logger = init_logger(__name__)
CPU_SYSFS = Path("/sys/devices/system/cpu")
NODE_SYSFS = Path("/sys/devices/system/node")
THREAD_STATUS = Path("/proc/thread-self/status")


def parse_id_list(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        result.update(range(int(first), int(last or first) + 1))
    return result


def physical_cpus() -> list[int]:
    chosen: dict[tuple[str, ...], int] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = CPU_SYSFS / f"cpu{cpu}" / "topology"
        try:
            key = tuple(
                (root / name).read_text().strip()
                for name in ("physical_package_id", "core_id")
            )
        except OSError:
            logger.warning_once(
                "Physical core topology unavailable; using allowed logical CPUs."
            )
            # Missing topology must not widen a container/taskset CPU mask.
            key = (str(cpu),)
        chosen.setdefault(key, cpu)
    return list(chosen.values())


@dataclass(frozen=True)
class NumaNode:
    node_id: int
    cpus: tuple[int, ...]


def cpu_numa_nodes() -> tuple[NumaNode, ...]:
    cpus = set(physical_cpus())
    nodes = []
    try:
        for node in sorted(parse_id_list((NODE_SYSFS / "online").read_text())):
            local = cpus & parse_id_list(
                (NODE_SYSFS / f"node{node}" / "cpulist").read_text()
            )
            if local:
                nodes.append(NumaNode(node, tuple(sorted(local))))
    except (OSError, ValueError):
        nodes = []
    if {cpu for node in nodes for cpu in node.cpus} != cpus:
        logger.warning_once(
            "CPU NUMA topology is unavailable/incomplete; using the allowed "
            "physical CPUs without NUMA sharding."
        )
        return (NumaNode(-1, tuple(sorted(cpus))),)
    return tuple(nodes)


def host_memory_nodes() -> tuple[int, ...]:
    """Nodes for automatic interleave, only when no memory policy is explicit."""
    try:
        allowed = next(
            parse_id_list(line.split(":", 1)[1])
            for line in THREAD_STATUS.read_text().splitlines()
            if line.startswith("Mems_allowed_list:")
        )
    except (OSError, ValueError, StopIteration):
        return ()
    candidates = allowed
    if len(candidates) > 1 and (libnuma := get_libnuma()) is not None:
        mode = ctypes.c_int()
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask = (ctypes.c_ulong * max(16, (max(allowed) + bits) // bits))()
        rc = libnuma.get_mempolicy(
            ctypes.byref(mode), mask, ctypes.c_ulong(len(mask) * bits + 1), None, 0
        )
        if rc:
            logger.warning_once(
                "Cannot read NUMA memory policy; leaving shared host placement "
                "unchanged. Check container memory-policy permissions."
            )
            return ()
        # A VMA policy overrides the thread policy, including bind restrictions
        # and interleave masks. Leave all explicit modes and flags inherited.
        if mode.value != 0:  # MPOL_DEFAULT
            return ()
    return tuple(sorted(candidates))
