# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in IMMA experiments using ExLlamaV3's pinned private MoE ABI."""

from benchmarks.kernels.exl3_m32.launcher import Launcher as RowsLauncher


class Launcher(RowsLauncher):
    variants = ("m32_predicated", "int8", "int8_residual")

    def __init__(self, extension, library, variant="int8"):
        super().__init__(extension, library, variant)
