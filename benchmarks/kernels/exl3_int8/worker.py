# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install INT8 experts on SM80 for controlled model experiments."""

from kernels.exl3_m32.worker import Exl3RowsWorkerExtension

from benchmarks.kernels.exl3_int8.launcher import Launcher


class Exl3Int8WorkerExtension(Exl3RowsWorkerExtension):
    launcher_class = Launcher
