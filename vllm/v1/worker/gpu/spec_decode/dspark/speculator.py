# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch
from torch.profiler import record_function

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.markov_argmax import (
    FusedMarkovSampler,
    build_fused_markov_sampler,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        self.hidden_states = None
        self._target_hidden_states_workspace: torch.Tensor | None = None

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32
        )
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification
        )

        # Fused Markov chain (greedy vocab-sharded path only); built in
        # load_draft_model once the head's weights and shard geometry exist.
        self._fused_markov: FusedMarkovSampler | None = None

    def set_target_hidden_states_workspace(self, workspace: torch.Tensor) -> bool:
        if (
            workspace.is_contiguous()
            and workspace.dtype == self.dtype
            and workspace.device == self.device
        ):
            self._target_hidden_states_workspace = workspace.view(-1)
            return True
        return False

    def _prepare_context_hidden_states(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_target_tokens: int,
    ) -> torch.Tensor:
        if not aux_hidden_states:
            return last_hidden_states[:num_target_tokens]

        num_rows = aux_hidden_states[0].shape[0]
        packed_width = sum(tensor.shape[-1] for tensor in aux_hidden_states)
        packed_numel = num_rows * packed_width
        workspace = self._target_hidden_states_workspace
        can_reuse_workspace = (
            workspace is not None
            and workspace.numel() >= packed_numel
            and all(
                tensor.untyped_storage().data_ptr()
                != workspace.untyped_storage().data_ptr()
                for tensor in aux_hidden_states
            )
        )
        if can_reuse_workspace:
            assert workspace is not None
            packed_hidden_states = workspace[:packed_numel].view(num_rows, packed_width)
            torch.cat(aux_hidden_states, dim=-1, out=packed_hidden_states)
        else:
            packed_hidden_states = torch.cat(aux_hidden_states, dim=-1)

        hidden_states = self.model.combine_hidden_states(packed_hidden_states)
        return hidden_states[:num_target_tokens]

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        if self.enable_adaptive_verification and model.model.confidence_head is None:
            raise ValueError(
                "Adaptive verification needs a DSpark checkpoint with a confidence "
                "head, and this one has none. Pass "
                "enable_adaptive_verification=false in the speculative config to verify"
                " a fixed number of drafts instead."
            )
        # `draft_logits is None` is greedy drafting, which is the only mode the
        # fused step covers -- it reduces the logit row instead of emitting it.
        # `_validate_local_argmax_reduction` rejects the other combination
        # outright, but it runs after this method, so the condition is spelled
        # out here rather than assumed.
        if (
            self.use_local_argmax_reduction
            and self.draft_logits is None
            and envs.VLLM_DSPARK_FUSED_MARKOV
        ):
            self._fused_markov = build_fused_markov_sampler(
                model, self.max_num_reqs, self.num_speculative_steps, self.device
            )
        return model

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position Q; the target verifies
        # it with the predecessor's Gumbel key (Q-1). Pass Q-1.
        return gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _validate_local_argmax_reduction(self) -> None:
        # DSpark's greedy selection spans the SUM of two vocab-parallel heads
        # (base draft logits + the Markov transition bias), so it needs a
        # different pair of hooks than the base class's single get_top_tokens.
        if not self.use_local_argmax_reduction:
            return
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "use_local_argmax_reduction is not compatible with "
                "draft_sample_method='probabilistic': verification needs whole "
                "processed logit rows, which is exactly what the reduction "
                "avoids materializing."
            )
        missing = [
            name
            for name in ("compute_draft_logits_shard", "select_draft_token_shard")
            if not hasattr(self.model, name)
        ]
        if missing:
            raise ValueError(
                "use_local_argmax_reduction is enabled but DSpark draft model "
                f"{self.model.__class__.__name__} does not implement "
                f"{', '.join(missing)}()."
            )
        logger.info(
            "DSpark: vocab-sharded Markov head with local argmax reduction "
            "(communication: O(2*tp_size) vs O(vocab_size) per draft position)."
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        # Under the local-argmax reduction these are this rank's vocab columns
        # only, so `vocab_size` below is the shard width -- nothing in this
        # method reads it as a vocab id.
        with record_function("dspark::draft_logits"):
            if self.use_local_argmax_reduction:
                base_logits = self.model.compute_draft_logits_shard(sample_hidden)
            else:
                base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        if self._fused_markov is not None:
            # Same chain as the loop below, with each step's seven kernels and
            # three [num_reqs, shard_width] round trips collapsed into one
            # pass over markov_w2. Writes draft_tokens directly.
            self._fused_markov.sample(num_reqs, base_logits, prev, self.draft_tokens)
            return

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            if self.use_local_argmax_reduction:
                # Greedy-only (enforced in _validate_local_argmax_reduction):
                # the argmax over base + bias reduces to one (value, id) pair
                # per rank, so neither head materializes a full-vocab row.
                draft_sampled_i = self.model.map_draft_to_target(
                    self.model.select_draft_token_shard(markov_embed, base_logits[:, i])
                )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
                prev = draft_sampled_i
                continue
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        # NOTE: these scopes are host-side, so they only resolve when the step
        # runs eagerly. Under CUDA-graph replay the whole step is one launch and
        # the ranges do not reappear -- profile with cudagraphs off to use them.
        with record_function("dspark::draft_backbone"):
            head_hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        with record_function("dspark::markov_sampling"):
            self._sample_sequential(num_reqs, head_hidden)
