# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size
        extra = self.vllm_config.additional_config
        cpu_hybrid = (
            isinstance(extra, dict)
            and extra.get("ced_prefill", False)
            and "deepseek_v41_hybrid" in extra
        )
        model = self.vllm_config.model_config
        exl3_moe = (
            envs.VLLM_EXL3_PP_DECODE_BATCHING
            and model.quantization == "exl3"
            and (
                model.hf_config.model_type in ("glm5_next", "glm5_next_text")
                or model.hf_text_config.model_type == "qwen4_exp_text"
            )
        )
        self.align_hybrid_decodes = (
            self.pp_size > 1 and self.use_v2_model_runner and (cpu_hybrid or exl3_moe)
        )
        if exl3_moe and self.align_hybrid_decodes:
            logger.info("Batching EXL3 pipeline decodes after output fences.")
        self._decode_phase: int | None = None

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if self.align_hybrid_decodes:
            decodes = [r for r in self.running if not r.is_prefill_chunk]
            if not decodes:
                self._decode_phase = None
            else:
                if self._decode_phase is None:
                    self._decode_phase = (
                        min(r.next_decode_eligible_step for r in decodes) % self.pp_size
                    )
                # Join one decode batch after the PP output is available. Moving
                # eligibility forward preserves the sampled-token ring's fence.
                for request in decodes:
                    request.next_decode_eligible_step += (
                        self._decode_phase - request.next_decode_eligible_step
                    ) % self.pp_size
        return super().schedule(throttle_prefills)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders
            request.spec_token_ids_step_id = (
                scheduler_output.scheduler_step_id
                if (
                    self.use_v2_model_runner
                    and request.use_structured_output
                    and self._spec_token_placeholders
                )
                else None
            )

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
