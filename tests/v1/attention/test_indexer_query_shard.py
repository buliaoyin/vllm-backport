"""CPU tests for indexer query-sharding: the 2a partition, its skip_kv_gather
rule, and the 2b Q-path row ranges that must compose with it.

Runs without a GPU on purpose: both regressions guarded here are silent (wrong
top-k indices, no crash), so their guards must not depend on scarce hardware.
Imports the shipped modules directly -- an earlier revision tested a private
copy of the helpers, which says nothing about the tree once the two diverge.
"""

from typing import Any, cast

from vllm.distributed.utils import balanced_row_bounds
from vllm.v1.attention.backends.mla.indexer import (
    MIN_SHARD_TOKENS,
    DeepseekV32IndexerPrefillChunkMetadata,
    ShardedChunkSpec,
    indexer_q_row_ranges,
    indexer_shard_size_for_batch,
    shard_chunk_specs_by_query,
)

FAILURES: list[str] = []


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


def _chunk(
    token_start: int, token_end: int, shard_row_counts: list[int] | None
) -> DeepseekV32IndexerPrefillChunkMetadata:
    """A chunk carrying only the fields the Q-path row ranges depend on.

    Built from the real dataclass rather than a stand-in so a renamed field
    fails here instead of passing against a mock.
    """
    none = cast(Any, None)
    return DeepseekV32IndexerPrefillChunkMetadata(
        block_table=none,
        cu_seqlen_ks=none,
        cu_seqlen_ke=none,
        cu_seq_lens=none,
        token_to_seq=none,
        total_seq_lens=token_end - token_start,
        token_start=token_start,
        token_end=token_end,
        num_reqs=1,
        shard_row_counts=shard_row_counts,
    )


def test_partition_is_exact():
    """Union over ranks is the original range: no gap, no overlap, any count."""
    for n in (0, 1, 7, 8, 9, 15, 17, 1023, 2048, 8192):
        for tp in (1, 2, 4, 8):
            for base in (0, 5, 4096):
                rows: list[int] = []
                for r in range(tp):
                    lo, hi = balanced_row_bounds(base, base + n, r, tp)
                    check(lo <= hi, f"inverted bounds n={n} tp={tp} r={r}")
                    rows.extend(range(lo, hi))
                check(
                    rows == list(range(base, base + n)),
                    f"partition not exact: n={n} tp={tp} base={base}",
                )
                counts = [
                    balanced_row_bounds(base, base + n, r, tp)[1]
                    - balanced_row_bounds(base, base + n, r, tp)[0]
                    for r in range(tp)
                ]
                check(
                    max(counts) - min(counts) <= 1,
                    f"imbalance >1 row: n={n} tp={tp} counts={counts}",
                )


def test_first_subchunk_per_rank_gathers_kv():
    """THE REGRESSION GUARD.

    Every rank's first sub-chunk for a request group must gather K. Under the
    shipped rule (`skip_kv_gather = query_slice.start > 0`) ranks 1..7 start at
    a nonzero row and would skip the gather, reading an unpopulated workspace.
    This asserts the opposite, so it fails against the unfixed logic.
    """
    specs = [(slice(0, 4), slice(0, 8192))]
    for tp in (2, 4, 8):
        for rank in range(tp):
            sharded = shard_chunk_specs_by_query(specs, rank, tp)
            check(len(sharded) == 1, f"expected 1 sub-chunk, got {len(sharded)}")
            qs, skip = sharded[0].query_slice, sharded[0].skip_kv_gather
            check(
                skip is False,
                f"rank {rank}/{tp} skips the K gather on its first sub-chunk "
                f"(rows {qs.start}:{qs.stop}) -- reads an unpopulated workspace",
            )
            if rank > 0:
                check(
                    qs.start > 0,
                    "test is vacuous unless rank>0 starts at a nonzero row",
                )


def test_later_subchunks_reuse_the_gather():
    """Within one request group, only the rank's first sub-chunk gathers."""
    specs = [(slice(0, 2), slice(0, 4096)), (slice(0, 2), slice(4096, 8192))]
    for tp in (2, 8):
        for rank in range(tp):
            sharded = shard_chunk_specs_by_query(specs, rank, tp)
            flags = [s.skip_kv_gather for s in sharded]
            check(flags == [False, True], f"tp={tp} rank={rank} flags={flags}")


def test_new_request_group_regathers():
    """A different request group means different K: the gather must repeat."""
    specs = [(slice(0, 2), slice(0, 4096)), (slice(2, 4), slice(0, 4096))]
    for tp in (2, 8):
        for rank in range(tp):
            sharded = shard_chunk_specs_by_query(specs, rank, tp)
            flags = [s.skip_kv_gather for s in sharded]
            check(
                flags == [False, False],
                f"tp={tp} rank={rank} flags={flags}: second request group "
                "must re-gather, its K workspace differs",
            )


def test_ragged_and_tiny_ranges():
    """T % tp != 0, and ranges too small for every rank to own a row."""
    specs = [(slice(0, 1), slice(0, 8195))]  # 8195 = 8*1024 + 3
    seen: list[int] = []
    for rank in range(8):
        sharded = shard_chunk_specs_by_query(specs, rank, 8)
        check(len(sharded) == 1, f"rank {rank}: expected 1 sub-chunk")
        qs = sharded[0].query_slice
        seen.extend(range(qs.start, qs.stop))
    check(seen == list(range(8195)), "ragged partition not exact")

    # Fewer rows than ranks: the sub-chunk must stay REPLICATED on every rank
    # (counts None), never sharded with empty shards on some ranks -- a
    # rank-dependent chunk list desynchronises the per-chunk all-gather.
    tiny = [(slice(0, 1), slice(0, 3))]
    for r in range(8):
        sh = shard_chunk_specs_by_query(tiny, r, 8)
        check(len(sh) == 1, f"rank {r}: tiny sub-chunk missing from chunk list")
        check(
            sh[0].shard_row_counts is None and sh[0].query_slice == slice(0, 3),
            f"rank {r}: tiny sub-chunk must stay replicated, got {sh[0]}",
        )


def test_gatherv_sizes_match_the_partition():
    """The sizes carried on each spec must equal what each rank produces."""
    for n in (8192, 8195, 8):
        specs = [(slice(0, 1), slice(0, n))]
        counts = shard_chunk_specs_by_query(specs, 0, 8)[0].shard_row_counts
        actual = []
        for r in range(8):
            sh = shard_chunk_specs_by_query(specs, r, 8)
            check(
                sh[0].shard_row_counts == counts,
                f"n={n}: rank {r} disagrees on the gather sizes",
            )
            check(
                sh[0].gather_start == 0,
                f"n={n}: rank {r} gather_start {sh[0].gather_start} != 0",
            )
            actual.append(sh[0].query_slice.stop - sh[0].query_slice.start)
        check(counts == actual, f"n={n}: sizes {counts} != produced {actual}")
        check(sum(counts) == n, f"n={n}: sizes sum to {sum(counts)}")


def test_chunk_list_is_rank_uniform():
    """Every rank must emit the same chunk list shape for the same specs.

    The per-chunk top-k all-gather is entered once per chunk carrying
    shard_row_counts; if that set were data-dependent per rank (as it was when
    empty shards were dropped), the collective would hang on ragged tails.
    """
    for n in (1, 3, 7, 8, 9, 8195):
        specs = [
            (slice(0, 1), slice(0, 4096)),
            (slice(0, 1), slice(4096, 4096 + n)),
        ]
        shapes = {
            (
                len(sh),
                tuple(s.shard_row_counts is None for s in sh),
            )
            for sh in (shard_chunk_specs_by_query(specs, r, 8) for r in range(8))
        }
        check(len(shapes) == 1, f"n={n}: rank-dependent chunk list: {shapes}")


def test_short_prefills_fall_back_to_replicated():
    """The threshold is a wired function, not a documented intention.

    Note what is deliberately absent: there is no decode condition. A mixed
    prefill+decode batch still shards, which is correct for 2a because its
    chunks name only prefill rows. (The 2b Q path does decline such batches --
    see test_q_path_declines_decode_batches -- because the decode branch reads
    rows no chunk names.)
    """
    check(indexer_shard_size_for_batch(8192, 8) == 8, "8K prefill must shard")
    check(
        indexer_shard_size_for_batch(MIN_SHARD_TOKENS, 8) == 8,
        "at threshold must shard",
    )
    check(
        indexer_shard_size_for_batch(MIN_SHARD_TOKENS - 1, 8) == 1,
        "below threshold must fall back to replicated",
    )
    check(indexer_shard_size_for_batch(8192, 1) == 1, "tp=1 must not shard")
    # The fallback mechanism is "shard size collapses to 1", so the partition
    # at size 1 has to reproduce the replicated flags exactly.
    specs = [(slice(0, 2), slice(0, 4096)), (slice(0, 2), slice(4096, 8192))]
    got = shard_chunk_specs_by_query(specs, 0, indexer_shard_size_for_batch(1024, 8))
    want = [ShardedChunkSpec(r, q, q.start > 0, None, None) for r, q in specs]
    check(got == want, f"fallback diverges from the replicated path: {got}")


def test_the_regression_guard_can_actually_fail():
    """Meta-test: a guard that cannot fail protects nothing.

    Re-runs the guard against the rule the tree shipped before 2a
    (`skip_kv_gather = query_slice.start > 0`) and asserts it rejects it. If a
    future refactor makes the guard vacuous, this fails instead of going quiet.
    """

    def shipped_rule(chunk_specs, tp_rank, tp_size):
        out = []
        for req_slice, query_slice in chunk_specs:
            lo, hi = balanced_row_bounds(
                query_slice.start, query_slice.stop, tp_rank, tp_size
            )
            if hi <= lo:
                continue
            out.append(ShardedChunkSpec(req_slice, slice(lo, hi), lo > 0, None, None))
        return out

    global shard_chunk_specs_by_query
    real = shard_chunk_specs_by_query
    saved = list(FAILURES)
    try:
        shard_chunk_specs_by_query = shipped_rule
        FAILURES.clear()
        test_first_subchunk_per_rank_gathers_kv()
        fired = len(FAILURES)
    finally:
        shard_chunk_specs_by_query = real
        FAILURES.clear()
        FAILURES.extend(saved)
    check(fired > 0, "regression guard is vacuous: it accepts the shipped buggy rule")


def test_tp1_matches_shipped_behaviour():
    """At tp_size=1 the function must reproduce today's flags exactly."""
    specs = [(slice(0, 2), slice(0, 4096)), (slice(0, 2), slice(4096, 8192))]
    got = shard_chunk_specs_by_query(specs, 0, 1)
    want = [ShardedChunkSpec(r, q, q.start > 0, None, None) for r, q in specs]
    check(got == want, f"tp=1 diverges from shipped: {got} != {want}")


# ---------------------------------------------------------------------------
# Stage 2b: the Q path (wq_b + fused RoPE/quant) over this rank's rows only.
# ---------------------------------------------------------------------------


def _rank_chunks(
    specs: list[tuple[slice, slice]],
    group_base: dict[tuple[int, int], int],
    rank: int,
    tp: int,
) -> list[DeepseekV32IndexerPrefillChunkMetadata]:
    """Chunk metadata as the builder produces it for one rank.

    Mirrors build_prefill_chunk_metadata: the chunk's global token bounds are
    the request group's first token plus the (rank-narrowed) query slice.
    """
    out = []
    for spec in shard_chunk_specs_by_query(specs, rank, tp):
        base = group_base[(spec.req_slice.start, spec.req_slice.stop)]
        qs = spec.query_slice
        chunk = _chunk(base + qs.start, base + qs.stop, spec.shard_row_counts)
        if spec.gather_start is not None:
            # The builder's formula: rank slice start in batch tokens minus
            # the rank's offset within the chunk = the chunk's absolute
            # first row, where the gathered rows are written back.
            chunk.gather_token_start = chunk.token_start - (
                qs.start - spec.gather_start
            )
        out.append(chunk)
    return out


def test_gather_start_is_batch_absolute_for_every_request_group():
    """THE CONCURRENT-BATCH REGRESSION GUARD.

    gather_token_start is where the all-gathered top-k rows are written into
    the batch-token-indexed buffer, so it must carry the request group's base
    offset. A group-relative value is correct for the FIRST request in a
    batch (base 0) -- every single-request bench passes -- and silently
    corrupts every later request's indices under concurrency, which showed up
    as a 0.97 -> 0.85 gsm8k drop at num_concurrent=8.
    """
    specs = [(slice(0, 1), slice(0, 4096)), (slice(1, 2), slice(0, 4099))]
    base = {(0, 1): 0, (1, 2): 4096}
    for tp in (2, 8):
        for rank in range(tp):
            for chunk, spec in zip(
                _rank_chunks(specs, base, rank, tp),
                shard_chunk_specs_by_query(specs, rank, tp),
            ):
                if chunk.shard_row_counts is None:
                    continue
                prefix = sum(chunk.shard_row_counts[:rank])
                check(
                    chunk.gather_token_start == chunk.token_start - prefix,
                    f"tp={tp} rank={rank}: gather start "
                    f"{chunk.gather_token_start} != token_start - prefix "
                    f"({chunk.token_start} - {prefix}); a group-relative "
                    "value would write the second request's top-k at the "
                    "wrong buffer rows",
                )
                check(
                    chunk.gather_token_start
                    == base[(spec.req_slice.start, spec.req_slice.stop)]
                    + spec.gather_start,
                    f"tp={tp} rank={rank}: gather start must be group base + "
                    "pre-shard chunk start",
                )


def test_q_rows_are_exactly_the_rows_the_chunk_loop_reads():
    """THE COMPOSITION GUARD (2a x 2b).

    2b computes only some rows of q_quant; 2a's chunk loop reads
    q_quant[chunk.token_start : chunk.token_end]. Every read row must have been
    computed, and across ranks the reads must still cover the whole prefill.
    Rows are global throughout -- neither stage re-slices the other's output.
    """
    cases = [
        # (specs, group_base, num_tokens, all prefill rows)
        ([(slice(0, 1), slice(0, 8192))], {(0, 1): 0}, 8192),
        ([(slice(0, 1), slice(0, 8195))], {(0, 1): 0}, 8195),  # ragged
        ([(slice(0, 1), slice(0, 3))], {(0, 1): 0}, 3),  # fewer rows than ranks
        (  # two request groups, second offset by the first's length
            [(slice(0, 1), slice(0, 4096)), (slice(1, 2), slice(0, 4099))],
            {(0, 1): 0, (1, 2): 4096},
            8195,
        ),
        (  # one group, sub-chunked by the logits budget
            [(slice(0, 2), slice(0, 4096)), (slice(0, 2), slice(4096, 8192))],
            {(0, 2): 0},
            8192,
        ),
    ]
    for specs, base, num_tokens in cases:
        covered: set[int] = set()
        for tp in (2, 4, 8):
            union: set[int] = set()
            for rank in range(tp):
                chunks = _rank_chunks(specs, base, rank, tp)
                ranges = indexer_q_row_ranges(chunks, 0, num_tokens)
                # None = the caller runs the replicated Q path and computes
                # EVERY row -- always safe, and what mixed batches (a
                # replicated tiny sub-chunk among sharded ones) must get.
                computed = (
                    set(range(num_tokens))
                    if ranges is None
                    else {t for lo, hi in ranges for t in range(lo, hi)}
                )
                read = {t for c in chunks for t in range(c.token_start, c.token_end)}
                check(
                    read <= computed,
                    f"tp={tp} rank={rank}: {len(read - computed)} rows are read "
                    "by the chunk loop but never computed by the Q path",
                )
                union |= read
            check(
                union == set(range(num_tokens)),
                f"tp={tp}: ranks together read {len(union)} of {num_tokens} rows",
            )
            covered |= union
        check(covered == set(range(num_tokens)), "coverage lost across tp sizes")


def test_the_composition_guard_can_actually_fail():
    """Meta-test: double-slicing must be detectable.

    The failure this campaign warned about is applying the row partition twice
    -- once when 2a cut the chunk bounds and again when 2b picks its rows. That
    produces wrong top-k indices with no crash. Substituting exactly that
    mistake must make the composition guard fire.
    """

    def double_sliced(chunks, num_decodes, num_tokens, tp=8):
        out = []
        for c in chunks:
            lo, hi = balanced_row_bounds(c.token_start, c.token_end, 1, tp)
            if hi > lo:
                out.append((lo, hi))
        return out or None

    global indexer_q_row_ranges
    real = indexer_q_row_ranges
    saved = list(FAILURES)
    try:
        indexer_q_row_ranges = double_sliced
        FAILURES.clear()
        test_q_rows_are_exactly_the_rows_the_chunk_loop_reads()
        fired = len(FAILURES)
    finally:
        indexer_q_row_ranges = real
        FAILURES.clear()
        FAILURES.extend(saved)
    check(fired > 0, "composition guard is vacuous: it accepts a double-sliced Q path")


def test_q_path_declines_when_2a_ran_replicated():
    """No shard_row_counts means the chunks name every row, not this rank's.

    This is what makes the Q path inherit 2a's fallbacks (tp=1, DCP, PCP,
    short prefill) without re-deriving any of them.
    """
    chunks = [_chunk(0, 8192, None)]
    check(
        indexer_q_row_ranges(chunks, 0, 8192) is None,
        "must decline when the batch was not sharded by 2a",
    )
    mixed = [_chunk(0, 4096, [2048, 2048]), _chunk(4096, 8192, None)]
    check(
        indexer_q_row_ranges(mixed, 0, 8192) is None,
        "one unsharded chunk must decline the whole batch",
    )


def test_q_path_declines_decode_batches():
    """Decode reads q_quant[:num_decode_tokens] and weights[:batch*next_n] --
    rows outside every chunk's range, and the latter can run past the decode
    region entirely."""
    chunks = [_chunk(8, 8200, [1024] * 8)]
    check(
        indexer_q_row_ranges(chunks, 1, 8200) is None,
        "a batch with decode requests must run the replicated Q path",
    )
    check(
        indexer_q_row_ranges(chunks, 0, 8200) == [(8, 8200)],
        "the same batch without decode must shard",
    )


def test_q_path_declines_rows_past_the_buffer():
    """A chunk naming a row beyond num_tokens would silently truncate the
    output slice instead of raising."""
    chunks = [_chunk(0, 8193, [1024] * 8)]
    check(
        indexer_q_row_ranges(chunks, 0, 8192) is None,
        "must decline rather than write a truncated slice",
    )


def test_q_ranges_merge_adjacent_chunks():
    """Adjacent chunk ranges become one GEMM; disjoint ones stay separate."""
    counts = [1024] * 8
    adjacent = [_chunk(0, 1024, counts), _chunk(1024, 2048, counts)]
    check(
        indexer_q_row_ranges(adjacent, 0, 8192) == [(0, 2048)],
        "touching chunk ranges must merge into a single wq_b call",
    )
    disjoint = [_chunk(0, 1024, counts), _chunk(2048, 3072, counts)]
    check(
        indexer_q_row_ranges(disjoint, 0, 8192) == [(0, 1024), (2048, 3072)],
        "a gap between chunks must not be filled in",
    )
    empty = [_chunk(512, 512, counts)]
    check(
        indexer_q_row_ranges(empty, 0, 8192) is None,
        "a rank owning no rows must fall back rather than compute nothing",
    )


def test_full_size_output_buffers_satisfy_the_fused_op_contract():
    """The Q path hands `fused_indexer_q_rope_quant` slices of full-size
    buffers instead of letting it allocate compact ones. That only works if a
    row slice meets everything the op does to its output buffers: shape asserts
    against the (equally sliced) input, a `.view(torch.uint8)` reinterpret of
    the FP8 values, and writes that land in the parent rows and nowhere else.
    """
    import torch

    num_tokens, n_head, head_dim = 32, 4, 128
    q_quant = torch.zeros((num_tokens, n_head, head_dim), dtype=torch.float8_e4m3fn)
    weights = torch.zeros((num_tokens, n_head), dtype=torch.float32)
    # `indexer_weights` reaches the indexer as a column split of the merged
    # input GEMM, i.e. row-strided but unit-strided in the head dim.
    merged = torch.zeros((num_tokens, n_head + 8), dtype=torch.float32)
    indexer_weights = merged.split([n_head, 8], dim=-1)[0]

    lo, hi = 8, 20
    q = torch.zeros((hi - lo, n_head, head_dim), dtype=torch.bfloat16)
    q_out, w_out = q_quant[lo:hi], weights[lo:hi]

    check(q_out.shape == q.shape, "op asserts index_q_fp8.shape == index_q.shape")
    check(
        w_out.shape == indexer_weights[lo:hi].shape,
        "op asserts index_weights_out.shape == index_weights.shape",
    )
    check(q_out.is_contiguous(), "FP8 output slice must stay contiguous")
    check(
        w_out.stride(1) == 1,
        "kernel indexes weights_out with an implicit unit stride",
    )
    check(
        indexer_weights[lo:hi].stride(1) == 1,
        "kernel indexes index_weights with an implicit unit stride",
    )
    try:
        q_out.view(torch.uint8)
    except RuntimeError as e:  # pragma: no cover - the failure is the message
        check(False, f"op reinterprets the FP8 buffer as uint8 and cannot: {e}")

    w_out.fill_(3.5)
    written = [t for t in range(num_tokens) if float(weights[t, 0]) != 0.0]
    check(
        written == list(range(lo, hi)),
        f"writing the slice touched rows {written}, expected {lo}:{hi}",
    )


def test_the_query_shard_flag_defaults_off():
    """This fork defaults the shard OFF: on TP4 A6000 the gather cost measured
    -13% (upstream sm80 branch defaults it on for TP8). =1 must still opt in."""
    import os

    import vllm.envs as envs

    name = "VLLM_INDEXER_QUERY_SHARD"
    saved = os.environ.pop(name, None)
    try:
        check(getattr(envs, name) is False, f"{name} must default off (TP4 policy)")
        os.environ[name] = "1"
        check(getattr(envs, name) is True, f"{name} must read '1' as enabled")
    finally:
        os.environ.pop(name, None)
        if saved is not None:
            os.environ[name] = saved


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print(f"all {len(tests)} tests passed")
