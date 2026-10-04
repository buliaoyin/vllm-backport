# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.spec_decode.dynamic.adaptive import (
    uses_mtp_confidence_forecast,
    uses_scheduler_adaptive_verification,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    share_mtp_topk_indices: bool = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.dynamic_draft = uses_scheduler_adaptive_verification(vllm_config)
        self._active_draft_tokens = self.num_speculative_steps
        self._decode_managers: dict[int, SpeculatorCudaGraphManager] = {}
        self.record_scheduler_confidence = uses_mtp_confidence_forecast(vllm_config)
        self._collect_scheduler_confidence = False
        self.draft_token_confidence_probs = (
            torch.full_like(self.draft_tokens, float("nan"), dtype=torch.float32)
            if self.record_scheduler_confidence
            else None
        )
        self.scheduler_confidence_table = (
            torch.full_like(self.draft_tokens, float("nan"), dtype=torch.float32)
            if self.record_scheduler_confidence
            else None
        )

    @property
    def num_draft_tokens(self) -> int:
        return getattr(self, "_active_draft_tokens", self.num_speculative_steps)

    def set_draft_budget(self, drafts: int) -> None:
        if not self.dynamic_draft:
            return
        if not 1 <= drafts <= self.num_speculative_steps:
            raise ValueError("MTP draft budget exceeds the configured maximum")
        self._active_draft_tokens = drafts
        if self._decode_managers and drafts > 1:
            self.decode_cudagraph_manager = self._decode_managers[drafts]

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if not getattr(self, "_collect_scheduler_confidence", False):
            return super().sample_draft(
                hidden_states,
                sample_src_positions,
                idx_mapping,
                temperature,
                seeds,
                draft_step,
                draft_logits,
            )
        tokens, confidence = self.model.get_top_tokens_with_confidence(hidden_states)
        greedy = (idx_mapping >= 0) & (temperature[idx_mapping] == 0)
        confidence = torch.where(greedy, confidence, float("nan"))
        table = self.draft_token_confidence_probs
        assert table is not None
        rows = hidden_states.shape[0]
        table[:rows, 0].copy_(confidence)
        return tokens

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        if not self.dynamic_draft or not self.use_fused_multi_step_decode:
            return
        manager = self.decode_cudagraph_manager
        assert manager is not None
        self._decode_managers[self.num_speculative_steps] = manager
        for drafts in range(2, self.num_speculative_steps):
            self._decode_managers[drafts] = SpeculatorCudaGraphManager(
                self.vllm_config,
                self.device,
                manager.cudagraph_mode,
                decode_query_len=1,
                fixed_decode_query_len=True,
            )

    def capture(self) -> None:
        super().capture()
        try:
            for drafts in sorted(self._decode_managers, reverse=True):
                if drafts != self.num_speculative_steps:
                    self.set_draft_budget(drafts)
                    self._capture_decode()
        finally:
            self.set_draft_budget(self.num_speculative_steps)

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        if spec_config is not None and spec_config.mtp_token_map is not None:
            draft_model.configure_mtp_token_map(spec_config.mtp_token_map)
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        self._collect_scheduler_confidence = getattr(
            self, "record_scheduler_confidence", False
        )
        if self._collect_scheduler_confidence:
            assert self.draft_token_confidence_probs is not None
            self.draft_token_confidence_probs[:num_reqs, 1:].fill_(float("nan"))
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        self._collect_scheduler_confidence = False
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_draft_tokens > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        self._collect_scheduler_confidence = False
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
