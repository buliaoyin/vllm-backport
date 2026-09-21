# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.models.deepseek_v4_1.ced import CEDMetadata, ced_prefill_enabled
from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule
from vllm.triton_utils import tl, triton
from vllm.v1.metrics.stats import ExpertCacheStats
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.requires_eager_prefill = ced_prefill_enabled(vllm_config)
        self.ced_metadata = (
            CEDMetadata(vllm_config, device)
            if self.requires_eager_prefill and get_pp_group().is_last_rank
            else None
        )
        self.ced_step = None
        self.ced_disabled_requests: set[str] = set()
        from ..hybrid import hybrid_settings

        if hybrid_settings(vllm_config) is not None:
            self.hybrid_requests: dict[str, int | None] = {}
        self.hybrid_ready = False
        self.hybrid_active: str | None = None
        self.hybrid_generated = 0
        self.hybrid_steps = 0
        self.hybrid_peak_batch = 0
        self.hybrid_request_steps = 0
        self.cpu_expert_modules = [
            module for module in model.modules() if isinstance(module, CPUExpertModule)
        ]
        self.cpu_async_modules = [
            module
            for module in self.cpu_expert_modules
            if getattr(module.backend.config, "cuda_library_path", None)
        ]
        self._expert_cache_stats_enabled = False
        additional = vllm_config.additional_config
        self.cpu_phase_threads = (
            additional.get("cpu_phase_threads")
            if isinstance(additional, dict)
            else None
        )
        if self.cpu_phase_threads is not None and (
            len(self.cpu_phase_threads) != 2
            or any(not isinstance(n, int) or n < 1 for n in self.cpu_phase_threads)
        ):
            raise ValueError("cpu_phase_threads must be [prefill, decode] > 0")
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        self._set_cpu_phase_threads(input_batch.has_prefill)
        if (
            getattr(self, "hybrid_ready", False)
            and input_batch.num_reqs > self.hybrid_peak_batch
        ):
            self.hybrid_peak_batch = input_batch.num_reqs
            logger.info(
                "Hybrid active batch: %d requests, %d scheduled tokens, prefill=%s",
                input_batch.num_reqs,
                input_batch.num_tokens,
                input_batch.has_prefill,
            )
        for module in self.cpu_async_modules:
            module.hybrid_has_prefill = input_batch.has_prefill
        if getattr(self, "hybrid_requests", None):
            request = input_batch.req_ids[0]
            if self.hybrid_active is None:
                self.hybrid_active = request
                for module in self.cpu_async_modules:
                    if module.gpu_cache is not None:
                        module.gpu_cache.begin_request(self.hybrid_requests[request])
            for module in self.cpu_async_modules:
                if module.gpu_cache is not None:
                    feedback = module.gpu_cache.decode_feedback
                    if feedback is not None:
                        feedback.set_enabled(not input_batch.has_prefill)
            self.hybrid_decoding = not input_batch.has_prefill
            self.hybrid_decode_requests = input_batch.num_reqs
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self.requires_eager_prefill:
            model_inputs["ced_step"] = self.ced_step
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def _set_cpu_phase_threads(self, has_prefill: bool) -> None:
        phase_threads = getattr(self, "cpu_phase_threads", None)
        if phase_threads is None:
            return
        threads = phase_threads[0 if has_prefill else 1]
        changed = [
            module.backend
            for module in self.cpu_async_modules
            if module.backend.config.num_threads != threads
        ]
        if changed:
            torch.cuda.current_stream().synchronize()
            for backend in changed:
                backend.set_num_threads(threads)

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
        if self.cpu_async_modules:
            torch.cuda.current_stream().synchronize()
            for module in self.cpu_async_modules:
                module.backend.check_cuda_errors()
            if getattr(self, "hybrid_active", None) and self.hybrid_decoding:
                self.hybrid_generated += int(num_sampled.sum().item())
                self.hybrid_steps += 1
                self.hybrid_request_steps += self.hybrid_decode_requests
                self._refresh_decode_cache()

    def _refresh_decode_cache(self):
        # Each step visits one layer; each layer enforces its own minimum interval.
        module = self.cpu_async_modules[
            (self.hybrid_steps - 1) % len(self.cpu_async_modules)
        ]
        if module.gpu_cache is None:
            return
        remaining_steps = None
        if (
            self.hybrid_request_steps == self.hybrid_steps
            and len(self.hybrid_requests) == 1
            and self.hybrid_active is not None
        ):
            limit = self.hybrid_requests.get(self.hybrid_active)
            if limit is not None:
                outputs_per_step = max(1.0, self.hybrid_generated / self.hybrid_steps)
                remaining_steps = (
                    max(0, limit - 1 - self.hybrid_generated) / outputs_per_step
                )
        module.gpu_cache.refresh_decode(
            self.hybrid_steps, remaining_steps=remaining_steps
        )

    def reset_expert_cache_stats(self) -> None:
        self._expert_cache_stats_enabled = any(
            module.gpu_cache is not None for module in self.cpu_expert_modules
        )
        if self._expert_cache_stats_enabled:
            torch.cuda.current_stream().synchronize()
            for module in self.cpu_expert_modules:
                module.take_cache_stats()

    def take_expert_cache_stats(self) -> ExpertCacheStats | None:
        if not self._expert_cache_stats_enabled:
            return None
        stats = ExpertCacheStats()
        for module in self.cpu_expert_modules:
            stats.accumulate(module.take_cache_stats())
        return stats

    def add_request(self, req_index, new_req_data):
        if self.requires_eager_prefill:
            params = new_req_data.sampling_params
            if params is not None and params.prompt_logprobs is not None:
                self.ced_disabled_requests.add(new_req_data.req_id)
        if getattr(self, "hybrid_ready", False):
            params = new_req_data.sampling_params
            self.hybrid_requests[new_req_data.req_id] = (
                params.max_tokens if params is not None else None
            )
        return super().add_request(req_index, new_req_data)

    def _finish_hybrid_request(self):
        if self.hybrid_active is None:
            return
        rate = (
            self.hybrid_generated / self.hybrid_request_steps
            if self.hybrid_request_steps
            else None
        )
        results = [
            module.gpu_cache.finish_request(rate)
            for module in self.cpu_async_modules
            if module.gpu_cache is not None
        ]
        results = [item for item in results if item is not None]
        if results:
            hits = sum(item[0] for item in results)
            routes = sum(item[1] for item in results)
            logger.info(
                "Hybrid request %s: GPU expert hit %d/%d (%.2f%%), "
                "decode steps=%d, sampled tokens=%d",
                self.hybrid_active,
                hits,
                routes,
                100 * hits / max(1, routes),
                self.hybrid_steps,
                self.hybrid_generated,
            )
        self.hybrid_active = None
        self.hybrid_generated = self.hybrid_steps = self.hybrid_request_steps = 0

    def remove_request(self, req_id):
        self.ced_disabled_requests.discard(req_id)
        if self.ced_metadata is not None:
            self.ced_metadata.remove_request(req_id)
        model = getattr(self, "model", None)
        if model is not None:
            for module in model.modules():
                ced = getattr(module, "ced_prefill", None)
                if ced is not None:
                    ced.remove_request(req_id)
        if hasattr(self, "hybrid_requests"):
            self.hybrid_requests.pop(req_id, None)
            if not self.hybrid_requests:
                self._finish_hybrid_request()
        return super().remove_request(req_id)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
    ):
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
        )
        self.ced_step = None
        if self.ced_metadata is not None and not all(
            req_id in self.ced_disabled_requests for req_id in input_batch.req_ids
        ):
            self.ced_step = self.ced_metadata.prepare(
                input_batch,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                disabled_requests=self.ced_disabled_requests,
            )
        return metadata

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs
