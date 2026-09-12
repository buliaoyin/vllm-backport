# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary, teacher-forced diagnostics for EXL3 prefill arithmetic."""

from collections import Counter

import torch
from exl3_profile_worker import Exl3ProfileWorkerExtension


def distribution_metrics(reference, candidate, targets):
    """Return per-row divergences in nats, using float64 normalization/reduction."""
    lp = reference.double().log_softmax(-1)
    lq = candidate.double().log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    lm = torch.logaddexp(lp, lq) - 0.6931471805599453
    native_top1 = lp.argmax(-1)
    candidate_top1 = lq.argmax(-1)
    native_second = lp.scatter(-1, native_top1[:, None], float("-inf")).argmax(-1)
    candidate_second = lq.scatter(-1, candidate_top1[:, None], float("-inf")).argmax(-1)
    pt = lp.topk(2, dim=-1).values
    qt = lq.topk(2, dim=-1).values
    selected = targets[:, None]
    fields = {
        "kl_native_int8": (p * (lp - lq)).sum(-1),
        "kl_int8_native": (q * (lq - lp)).sum(-1),
        "js": ((p * (lp - lm)).sum(-1) + (q * (lq - lm)).sum(-1)) / 2,
        "tv": (p - q).abs().sum(-1) / 2,
        "max_probability_delta": (p - q).abs().amax(-1),
        "native_target_logprob": lp.gather(-1, selected).flatten(),
        "candidate_target_logprob": lq.gather(-1, selected).flatten(),
        "native_top1": native_top1,
        "candidate_top1": candidate_top1,
        "native_top2": native_second,
        "candidate_top2": candidate_second,
        "native_top1_probability": pt[:, 0].exp(),
        "candidate_top1_probability": qt[:, 0].exp(),
        "native_top1_margin": pt[:, 0] - pt[:, 1],
        "candidate_top1_margin": qt[:, 0] - qt[:, 1],
    }
    values = torch.stack([x.double() for x in fields.values()], dim=-1).cpu()
    result = []
    for row in values.tolist():
        item = dict(zip(fields, row))
        for key in ("native_top1", "candidate_top1", "native_top2", "candidate_top2"):
            item[key] = int(item[key])
        assert all(torch.isfinite(torch.tensor(row)))
        assert item["kl_native_int8"] >= -1e-10
        item["top1_equal"] = item["native_top1"] == item["candidate_top1"]
        result.append(item)
    return result


class Exl3DivergenceWorker(Exl3ProfileWorkerExtension):
    def install_divergence_probe(self):
        from vllm import envs
        from vllm.config import CompilationMode
        from vllm.model_executor.layers.quantization import exl3
        from vllm.model_executor.layers.quantization.utils import exl3_prefill

        runner = self.model_runner
        assert self.vllm_config.compilation_config.mode == CompilationMode.NONE
        assert self.vllm_config.model_config.enforce_eager
        assert not envs.VLLM_TOKEN_BUCKET_PAD
        assert self.vllm_config.parallel_config.tensor_parallel_size == 1
        self._dv_methods = []
        for module in self.get_model().modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, exl3.Exl3MoEMethod):
                assert method.prefill_workspace, method.prefix
                self._dv_methods.append((method, method.prefill_workspace))
        self._dv_phase = None
        self._dv_requests = {}
        self._dv_reference = {}
        self._dv_metrics = []
        self._dv_counts = Counter()
        self._dv_rows = Counter()
        self._dv_schedule = []
        self._dv_batch = None
        self._dv_forced = []
        self._dv_logits_info = {}

        for owner, name, label in (
            (exl3_prefill, "moe_int8", "int8"),
            (exl3, "_exl3_moe_fused", "native"),
            (exl3, "_exl3_moe_decode_int8", "decode_int8"),
            (exl3, "_exl3_moe_decode", "decode_native"),
        ):
            original = getattr(owner, name)

            def count(x, *args, _original=original, _label=label, **kwargs):
                self._dv_counts[_label] += 1
                self._dv_rows[f"{_label}:{x.shape[0]}"] += 1
                return _original(x, *args, **kwargs)

            setattr(owner, name, count)

        if runner.is_last_pp_rank:
            original_sample = runner.sample
            original_logits = self.get_model().compute_logits

            def logits(hidden):
                raw = original_logits(hidden)
                batch = self._dv_batch
                if self._dv_phase is None or batch is None:
                    return raw
                self._dv_logits_info = {
                    "dtype": str(raw.dtype),
                    "vocab_size": raw.shape[-1],
                }
                assert batch.num_tokens_after_padding == sum(batch.num_scheduled_tokens)
                positions = batch.positions[batch.logits_indices].cpu().tolist()
                assert len(positions) == len(batch.req_ids) == raw.shape[0]
                active = []
                for row, (req_id, pos) in enumerate(zip(batch.req_ids, positions)):
                    meta = self._dv_requests[req_id]
                    step = pos + 1 - meta["prompt_length"]
                    self._dv_schedule.append(
                        [
                            meta["name"],
                            step,
                            int(batch.num_scheduled_tokens[row]),
                            len(batch.req_ids),
                        ]
                    )
                    if 0 <= step < len(meta["targets"]):
                        active.append((row, meta, step))
                if self._dv_phase == "reference":
                    cpu = raw.detach().float().cpu()
                    for row, meta, step in active:
                        key = (meta["name"], step)
                        assert key not in self._dv_reference, key
                        self._dv_reference[key] = cpu[row].clone()
                elif active:
                    keys = [(meta["name"], step) for _, meta, step in active]
                    ref = torch.stack([self._dv_reference[key] for key in keys]).to(
                        device=raw.device
                    )
                    row_ids = torch.tensor(
                        [row for row, _, _ in active], device=raw.device
                    )
                    target_ids = torch.tensor(
                        [meta["targets"][step] for _, meta, step in active],
                        device=raw.device,
                    )
                    measured = distribution_metrics(ref, raw[row_ids], target_ids)
                    for item, (_, meta, step) in zip(measured, active):
                        item.update(
                            name=meta["name"], step=step, target=meta["targets"][step]
                        )
                        self._dv_metrics.append(item)
                self._dv_forced = active
                return raw

            def sample(hidden, batch, grammar):
                self._dv_batch = batch
                try:
                    result = original_sample(hidden, batch, grammar)
                    if self._dv_phase is not None and self._dv_forced:
                        output = result[0].sampled_token_ids
                        rows = torch.tensor(
                            [i for i, _, _ in self._dv_forced], device=output.device
                        )
                        tokens = torch.tensor(
                            [
                                meta["targets"][step]
                                for _, meta, step in self._dv_forced
                            ],
                            device=output.device,
                            dtype=output.dtype,
                        )
                        output[rows, 0] = tokens
                    return result
                finally:
                    self._dv_batch = None

            self.get_model().compute_logits = logits
            runner.sample = sample
        return self.get_exl3_runtime_state()

    def begin_divergence(self, phase, use_int8, requests, min_rows=9):
        torch.accelerator.synchronize()
        if phase == "reference":
            self._dv_reference.clear()
        self._dv_phase = phase
        self._dv_requests = requests
        self._dv_metrics = []
        self._dv_schedule = []
        self._dv_counts.clear()
        self._dv_rows.clear()
        for method, workspace in self._dv_methods:
            method.prefill_workspace = workspace if use_int8 else []
            method.prefill_min_rows = min_rows
        return {"phase": phase, "int8": use_int8, "min_rows": min_rows}

    def finish_divergence(self):
        torch.accelerator.synchronize()
        result = {
            "phase": self._dv_phase,
            "metrics": self._dv_metrics,
            "schedule": self._dv_schedule,
            "calls": dict(self._dv_counts),
            "rows": dict(self._dv_rows),
            "reference_rows": len(self._dv_reference),
            "logits": self._dv_logits_info,
        }
        self._dv_phase = None
        return result
