# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install row kernels on SM80 ranks for controlled, same-engine comparisons."""

import hashlib
from pathlib import Path

import torch
from exl3_profile_worker import Exl3ProfileWorkerExtension
from kernels.exl3_m32.launcher import Launcher


class Exl3RowsWorkerExtension(Exl3ProfileWorkerExtension):
    def configure_exl3_optimization(self, options):
        from vllm.model_executor.layers.quantization.exl3 import (
            Exl3MoEMethod,
            _extension,
        )

        extension = _extension()
        if not hasattr(self, "_rows_original_moe"):
            self._rows_original_moe = extension.exl3_moe
        extension.exl3_moe = self._rows_original_moe
        result = super().configure_exl3_optimization(options)
        library = options.get("rows_library")
        if not library or torch.cuda.get_device_capability() != (8, 0):
            result["row_kernel"] = "native"
            return result
        launcher = Launcher(
            extension, library, options.get("rows_variant", "m32_predicated")
        )
        methods = 0
        for module in self.get_model().modules():
            method = getattr(module, "quant_method", None)
            if not isinstance(method, Exl3MoEMethod):
                continue
            launcher.plan(method.ptrs[0].device.index, method.bits, method.flags)
            methods += 1
        extension.exl3_moe = launcher
        result["row_kernel"] = {
            "variant": launcher.variant,
            "library": launcher.library,
            "sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
            "expert_layers": methods,
            "plans": [
                {key: value for key, value in plan.items() if key != "locks"}
                for plan in launcher.plans.values()
            ],
        }
        return result
