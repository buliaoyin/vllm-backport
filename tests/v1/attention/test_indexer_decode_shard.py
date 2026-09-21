"""CPU tests for the decode half of the indexer query shard: the query-group
partition, the batch-absolute row offsets it writes at, its gate, and the
float32 reduction that reassembles the batch.

Runs without a GPU on purpose, like its prefill sibling
(test_indexer_query_shard.py): every regression guarded here is silent (wrong
top-k rows, or a rank-dependent decision that hangs the per-layer collective),
so the guards must not depend on scarce hardware. Imports the shipped helpers
directly -- a test that restates the arithmetic proves only that the test can
do arithmetic.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    indexer_decode_shard_bounds,
    indexer_decode_shard_rows,
    indexer_shard_is_eligible,
)

FAILURES: list[str] = []

# The standing DSv4 decode shape: 21 ratio-4 layers, index_topk 512, next_n 6
# (num_speculative_tokens=5), and on SM80 the flattening path, which hands the
# kernels one query group per token (next_n collapses to 1 there).
TOPK = 512
NEXT_N = 6
MIN_REQS = 4


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def teardown_function(function) -> None:
    """Make the collected failures fail the test under pytest.

    `check` accumulates so that one run reports every broken case, which the
    __main__ runner below prints in a batch. Without this hook a pytest run
    would collect these tests and pass them unconditionally.
    """
    failures, FAILURES[:] = list(FAILURES), []
    assert not failures, "\n".join(failures)


def _bounds(batch_size, rank, tp, num_decodes=None, min_reqs=MIN_REQS):
    return indexer_decode_shard_bounds(
        batch_size,
        batch_size if num_decodes is None else num_decodes,
        rank,
        tp,
        min_reqs,
    )


def test_partition_is_exact_over_query_groups():
    """Union over ranks is the batch: no gap, no overlap, at any group count.

    Exactness is what makes the sum-reduction legal: a group owned by two ranks
    would be added twice, and one owned by none would arrive as zeros -- both
    silent, since zero and a doubled index are valid-looking indices.
    """
    for batch in (8, 9, 16, 24, 132, 136, 133, 512):
        for tp in (2, 4, 8):
            groups: list[int] = []
            counts = []
            for r in range(tp):
                bounds = _bounds(batch, r, tp)
                check(bounds is not None, f"batch={batch} tp={tp} r={r} declined")
                if bounds is None:
                    continue
                lo, hi = bounds
                check(lo <= hi, f"inverted bounds batch={batch} tp={tp} r={r}")
                groups.extend(range(lo, hi))
                counts.append(hi - lo)
            check(
                groups == list(range(batch)),
                f"partition not exact: batch={batch} tp={tp}",
            )
            check(
                max(counts) - min(counts) <= 1,
                f"imbalance >1 group: batch={batch} tp={tp} counts={counts}",
            )


def test_rows_are_batch_absolute_and_tile_the_buffer():
    """THE OFFSET GUARD (canon rule 36).

    Each rank writes its top-k straight into topk_indices_buffer, which is
    indexed by batch token. So rank r's rows must start at its FIRST group's
    batch-absolute offset, not at 0, and the ranks' row ranges must tile
    [0, batch * next_n) exactly. A group-relative offset is right for rank 0 --
    every single-rank check passes -- and puts every later rank's indices on
    another request's rows.
    """
    for batch in (24, 132, 133, 512):
        for tp in (2, 8):
            for next_n in (1, NEXT_N):
                rows: list[int] = []
                for r in range(tp):
                    bounds = _bounds(batch, r, tp)
                    row_lo, row_hi = indexer_decode_shard_rows(bounds, batch, next_n)
                    assert bounds is not None
                    check(
                        row_lo == bounds[0] * next_n,
                        f"batch={batch} tp={tp} r={r}: rows start at {row_lo}, "
                        f"not at the batch-absolute {bounds[0] * next_n}",
                    )
                    check(
                        (row_hi - row_lo) % next_n == 0,
                        f"batch={batch} tp={tp} r={r}: row range is not a whole "
                        "number of query groups, so a request's next_n rows "
                        "would straddle two ranks",
                    )
                    rows.extend(range(row_lo, row_hi))
                check(
                    rows == list(range(batch * next_n)),
                    f"batch={batch} tp={tp} next_n={next_n}: rows do not tile "
                    "the decode region exactly",
                )
                if tp > 1:
                    check(
                        indexer_decode_shard_rows(_bounds(batch, 1, tp), batch, next_n)[
                            0
                        ]
                        > 0,
                        "test is vacuous unless rank 1 starts at a nonzero row",
                    )


def test_replicated_path_is_the_whole_buffer():
    """None bounds must reproduce today's full-batch slice exactly."""
    for batch in (1, 6, 132):
        for next_n in (1, NEXT_N):
            check(
                indexer_decode_shard_rows(None, batch, next_n) == (0, batch * next_n),
                f"replicated rows must be the whole decode region "
                f"(batch={batch}, next_n={next_n})",
            )


def test_gate_falls_back_to_replicated():
    """The gate is a wired function, not a documented intention.

    tp=1, DCP and PCP all reach this as shard_size == 1, because the builder
    collapses the shard size through indexer_shard_is_eligible rather than
    re-deriving the condition per half.
    """
    check(not indexer_shard_is_eligible(1, 1, False), "tp=1 must not be eligible")
    check(not indexer_shard_is_eligible(8, 2, False), "DCP must not be eligible")
    check(not indexer_shard_is_eligible(8, 1, True), "PCP must not be eligible")
    check(indexer_shard_is_eligible(8, 1, False), "TP=8 alone must be eligible")

    check(_bounds(132, 0, 1) is None, "shard_size 1 (tp=1/DCP/PCP) must decline")
    check(
        _bounds(132, 0, 8, num_decodes=MIN_REQS - 1) is None,
        "below the request threshold must decline",
    )
    check(
        _bounds(132, 0, 8, num_decodes=MIN_REQS) is not None,
        "at the request threshold must shard",
    )
    check(
        _bounds(132, 0, 8, min_reqs=0) is None,
        "min_reqs=0 is the opt-out and must decline",
    )
    # Fewer query groups than ranks: every rank must decline, so no rank
    # launches the decode kernels over an empty row range just to reach the
    # collective. This is reachable -- the native (non-flattening) path has one
    # group per request, so 4 <= C < 8 lands here at tp=8.
    for r in range(8):
        check(
            _bounds(5, r, 8, num_decodes=5) is None,
            f"rank {r}: 5 groups over 8 ranks must stay replicated",
        )


def test_the_decision_is_rank_uniform():
    """Every rank must agree on whether the collective happens.

    The gate reads only replicated batch metadata, so a rank-dependent answer
    is impossible by construction -- this pins it, because the failure mode is
    a hang rather than a wrong number.
    """
    for batch in (1, 5, 7, 8, 24, 132, 133):
        for num_decodes in (1, 3, 4, 22, 85):
            for tp in (2, 8):
                decisions = {
                    _bounds(batch, r, tp, num_decodes=num_decodes) is None
                    for r in range(tp)
                }
                check(
                    len(decisions) == 1,
                    f"batch={batch} decodes={num_decodes} tp={tp}: "
                    f"rank-dependent participation {decisions}",
                )


def test_partition_is_fixed_by_the_captured_batch_size():
    """The cudagraph story: bounds depend only on the captured row count.

    A full-cudagraph decode replays with the padded request count baked in, and
    at replay the real count is smaller. Once the gate passes, the partition
    must be a pure function of batch_size, or capture and replay would disagree
    about who owns which rows.
    """
    for batch in (24, 136, 512):
        for r in range(8):
            captured = _bounds(batch, r, 8, num_decodes=batch // NEXT_N)
            for real_decodes in (MIN_REQS, 7, 22, 85):
                check(
                    _bounds(batch, r, 8, num_decodes=real_decodes) == captured,
                    f"batch={batch} rank={r}: partition moved with the live "
                    f"request count ({real_decodes}) instead of the captured "
                    "batch size",
                )


def test_zero_lanes_reduction_is_bit_exact():
    """The reduction carries int32 indices through float32 without loss.

    Reproduces what the ranks do: every rank zero-fills the rows it does not
    own, writes its own rows, and the sums are added. Exercised on the real
    spread (canon rule 34) -- indices up to the 65,536 compressed positions of
    a 262,144-token context, the -1 "no token" sentinel, and 0, which is both a
    valid index and the value every non-owner contributes.
    """
    torch.manual_seed(0)
    batch, tp = 132, 8
    max_index = 262144 // 4
    reference = torch.randint(0, max_index, (batch * NEXT_N, TOPK), dtype=torch.int32)
    reference[:, -3:] = -1  # tail slots the top-k kernel leaves at the pre-fill
    reference[3, 0] = 0
    reference[7, 1] = max_index - 1

    total = torch.zeros((batch * NEXT_N, TOPK), dtype=torch.float32)
    for r in range(tp):
        bounds = _bounds(batch, r, tp)
        row_lo, row_hi = indexer_decode_shard_rows(bounds, batch, NEXT_N)
        scattered = torch.zeros((batch * NEXT_N, TOPK), dtype=torch.float32)
        scattered[row_lo:row_hi] = reference[row_lo:row_hi]
        total += scattered

    out = torch.empty_like(reference)
    out.copy_(total.clamp_(-1, max_index))
    check(
        torch.equal(out, reference),
        "the float32 sum-of-zeros reduction is not bit-exact on real indices",
    )
    check(
        int(total.max()) <= max_index and int(total.min()) >= -1,
        "the clamp changed a real index, so it is not a no-op on real values",
    )


def test_the_shipped_helper_reassembles_the_batch():
    """THE END-TO-END GUARD, on the shipped reduction rather than a restatement.

    Simulates all 8 ranks: each starts from the -1 pre-fill, has the top-k
    kernel fill only its own rows, and calls `_all_reduce_decode_topk` with the
    collective stubbed. Every rank must end holding the full batch, equal to
    what the replicated path computes. This is what catches a reduction that
    reads the kernel's output slice instead of the buffer (it would lose the
    pre-filled -1 tail slots), or that writes back at the wrong offset.
    """
    try:
        from vllm.model_executor.layers import sparse_attn_indexer as sai
    except ImportError as e:
        # The consumer module pulls in the compiled extension; the partition
        # tests above deliberately do not, so only this one skips.
        pytest.skip(f"sparse_attn_indexer needs the built extension: {e}")

    torch.manual_seed(0)
    batch, tp, next_n = 132, 8, 1  # SM80 flattening path: one group per token
    rows, max_index = batch * next_n, 262144 // 4
    reference = torch.randint(0, max_index, (rows, TOPK), dtype=torch.int32)
    reference[:, -3:] = -1  # slots the top-k kernel leaves at the pre-fill

    contributions: list[torch.Tensor] = []
    reduced: list[torch.Tensor] = []

    def fake_all_reduce(x):
        contributions.append(x.clone())
        return reduced[0] if reduced else torch.zeros_like(x)

    real = sai.tensor_model_parallel_all_reduce
    try:
        sai.tensor_model_parallel_all_reduce = fake_all_reduce
        for pass_no in range(2):
            outs = []
            for r in range(tp):
                bounds = _bounds(batch, r, tp)
                row_lo, row_hi = indexer_decode_shard_rows(bounds, batch, next_n)
                buf = torch.full((rows + 16, TOPK), -1, dtype=torch.int32)
                buf[row_lo:row_hi] = reference[row_lo:row_hi]
                sai._all_reduce_decode_topk(buf, rows, TOPK, row_lo, row_hi, max_index)
                outs.append(buf)
            if pass_no == 0:
                # Second pass replays with the real sum every rank would see.
                reduced.append(torch.stack(contributions).sum(0))
            else:
                for r, buf in enumerate(outs):
                    check(
                        torch.equal(buf[:rows], reference),
                        f"rank {r} did not end with the full batch",
                    )
                    check(
                        bool((buf[rows:] == -1).all()),
                        f"rank {r} wrote past the decode region",
                    )
    finally:
        sai.tensor_model_parallel_all_reduce = real


def test_the_offset_guard_can_actually_fail():
    """Meta-test: a guard that cannot fail protects nothing.

    Substitutes the group-relative row offset -- the exact mistake canon rule
    36 was earned on -- and asserts the offset guard rejects it.
    """
    global indexer_decode_shard_rows
    real = indexer_decode_shard_rows
    saved = list(FAILURES)
    try:

        def group_relative_rows(bounds, batch_size, next_n):
            if bounds is None:
                return 0, batch_size * next_n
            return 0, (bounds[1] - bounds[0]) * next_n

        indexer_decode_shard_rows = group_relative_rows
        FAILURES.clear()
        test_rows_are_batch_absolute_and_tile_the_buffer()
        fired = len(FAILURES)
    finally:
        indexer_decode_shard_rows = real
        FAILURES.clear()
        FAILURES.extend(saved)
    check(fired > 0, "offset guard is vacuous: it accepts group-relative rows")


def test_the_rank_uniformity_guard_can_actually_fail():
    """Meta-test: dropping the ranks that own nothing must be detectable.

    The prefill half shipped exactly this bug once (ranks with no rows dropped
    a chunk, desynchronising the collective). Substituting the same rule for
    the decode gate must make the uniformity and exactness guards fire.
    """
    global indexer_decode_shard_bounds
    real = indexer_decode_shard_bounds
    saved = list(FAILURES)
    try:

        def drops_empty_ranks(batch_size, num_decodes, rank, size, min_reqs):
            if size <= 1 or min_reqs <= 0 or num_decodes < min_reqs:
                return None
            base, rem = divmod(batch_size, size)
            lo = rank * base + min(rank, rem)
            hi = lo + base + (rank < rem)
            return None if hi <= lo else (lo, hi)

        indexer_decode_shard_bounds = drops_empty_ranks
        FAILURES.clear()
        test_the_decision_is_rank_uniform()
        fired = len(FAILURES)
    finally:
        indexer_decode_shard_bounds = real
        FAILURES.clear()
        FAILURES.extend(saved)
    check(
        fired > 0,
        "rank-uniformity guard is vacuous: it accepts a partition that drops "
        "the ranks owning no group",
    )


def test_the_decode_shard_threshold_defaults_to_the_priced_value():
    """Default is the priced crossover; 0 is the opt-out."""
    import os

    import vllm.envs as envs

    name = "VLLM_INDEXER_DECODE_SHARD_MIN_REQS"
    saved = os.environ.pop(name, None)
    try:
        check(getattr(envs, name) == MIN_REQS, f"{name} must default to {MIN_REQS}")
        os.environ[name] = "0"
        check(getattr(envs, name) == 0, f"{name} must read '0' as the opt-out")
        check(
            _bounds(132, 0, 8, min_reqs=getattr(envs, name)) is None,
            f"{name}=0 must reach the replicated path",
        )
    finally:
        os.environ.pop(name, None)
        if saved is not None:
            os.environ[name] = saved


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    skipped = 0
    for t in tests:
        try:
            t()
        except pytest.skip.Exception as e:
            skipped += 1
            print(f"SKIPPED {t.__name__}: {e}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print(f"all {len(tests) - skipped} tests passed ({skipped} skipped)")
