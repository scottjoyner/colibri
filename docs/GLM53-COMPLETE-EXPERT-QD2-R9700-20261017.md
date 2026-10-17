# GLM-5.3 complete-expert NVMe reads — physical QD2 benchmark

**Date:** October 17, 2026. **Machine:** Joyner R9700 / Samsung 9100 PRO Gen5 SSD.
**Checkpoint:** pinned `zai-org/GLM-5.3-Flash@46a050bec13002ad29ee795c16e61cf4fa3c4a9c`, verified 62-shard Colibri int4 safetensor conversion.
**Scope:** physical read-only shadow benchmark, not live inference or prefetch.

## Architecture correction

Earlier PRs #10 and #11 read **one 4 MiB up-projection tensor window**
per predicted expert. The actual converted GLM expert requires **six**
separate tensors, distributed within the safetensor shard:

| Tensor | Source dtype | Stored bytes |
|---|---|---:|
| down_proj.weight | U8 | 4,194,304 |
| down_proj.weight.qs | F32 | 524,288 |
| gate_proj.weight | U8 | 4,194,304 |
| gate_proj.weight.qs | F32 | 524,288 |
| up_proj.weight | U8 | 4,194,304 |
| up_proj.weight.qs | F32 | 524,288 |
| **Total / complete expert** | | **14,155,776 (13.5 MiB)** |

All **12,384** experts were indexed and checked for the six exact names,
dtypes, sizes, unique source identity and file bounds. The source revision,
converted shard inventories and mounted Gen5 disk were checked before reads.

The benchmark is a **two-reader concurrent expert** pipeline: each worker
reads its expert's six tensors sequentially with Linux `O_DIRECT`;
at most two complete experts may be read concurrently. Memory reservation
is bounded to two logical experts (**27 MiB**) and an isolated systemd
cgroup with `MemoryMax=384M`, `MemorySwapMax=0`, low CPU/IO priority.
Physical byte counters include extra alignment padding and are measured
directly from `/sys/block/nvme0n1/stat`. Reads neither alter model files
nor place expert weights in CPU/GPU caches.

## Full held-out physical A/B

Fitted fusion uses route-history runs 01–04. Validation run05 decides
whether to keep the second prediction with the frozen threshold 0.40.
Native run06-v2 is never consulted during nomination, only to score
held-out usefulness. Exactly the same **462 routing events** were run
twice, in **ungated→gated** and **gated→ungated** orders.

| Metric | Frozen fusion | Validation-gated fusion |
|---|---:|---:|
| Candidate nominations | 924 | 715 |
| Predictions useful | 47.294% | 49.930% |
| Native selected expert recall | **11.824%** | 9.659% |
| **Complete experts physically read** | **920** | **714** |
| `O_DIRECT` bytes per run | **13.0459 GB** | **10.1248 GB** |
| Relative SSD read volume | baseline | **22.39% less** |
| Complete expert p99, order A/B | **10.316 ms** | 10.821 ms |
| Complete expert p99, order B/A | **10.511 ms** | 10.668 ms |
| Two-expert step p99, order A/B | **10.552 ms** | 10.887 ms |
| Two-expert step p99, order B/A | **10.805 ms** | 10.912 ms |

**2.9211 GB of physical direct SSD reads saved per full 462-event
replay**. Both benchmark orders returned the same candidates, accuracy,
expert-read count and byte volumes. Whole-SSD sector delta matched
the recorded `O_DIRECT` bytes in both passes. Initial 32-event
physical canary also passed (0.9075 vs 0.5247 GB reads), but its
short-run p99 has high variance; use the full-run results.

**Do not misinterpret p99:** the full-expert p99 is now ~10–11 ms,
compared with ~2–3 ms for a single 4 MiB window. The 10 ms layer
deadline assumed by previous PR #11 is **not sufficient to guarantee**
complete expert availability, even before Vulkan upload/conversion
time. Native GLM traces have no timing metadata for actual router
lookahead slack. Therefore this does not establish any token/s gain.

## EOF safety and exact source integrity

Some safetensor source files end on a non-4 KiB boundary. A padded
`O_DIRECT` request would reach beyond the actual file EOF. We split
the read: the aligned, valid portion uses `O_DIRECT`; an exceptionally
small **<4096-byte read-only tail** uses a separate buffered file
descriptor. This preserves exact source bytes; model files are
**never padded or modified**.

The independent physical verifier loaded **all 59 expert tensor windows
at non-4 KiB shard EOF boundaries** using this path, then compared the
assembled bytes **byte-for-byte** with the exact tensor bytes from
normal read-only file access. **59/59 PASS**, zero source file size or
modification-time changes, with **45,613,056 O_DIRECT bytes** and
**139,032 read-only buffered tail bytes** (largest individual tail:
4,080 bytes). No model weights, activation traces or private sessions
are published.

The 462-event benchmark happened not to nominate any EOF-ending tensor,
so every measured data byte in its A/B consisted solely of
`O_DIRECT` requests. EOF correctness is independently covered.

## Scope, regression and next gates

Sources:
- `benchmarks/joyner-glm53-20261016/benchmark_complete_expert_qd2.py`
- `benchmarks/joyner-glm53-20261016/verify_full_expert_eof.py`
- `c/tests/test_glm53_complete_expert_io.py`
- `benchmarks/joyner-glm53-20261016/hardware-20261017/complete-expert/`
  includes full A/B + reverse B/A, 32-event pilot, 59-window EOF
  verification, machine SHA256 evidence summary.

Five new aligned-span/EOF/no-overread tests join the existing
six bounded-reader, four trace-leakage and seven shadow planner
tests: **22 focused tests** on x1-370.

**Unresolved:**
1. Capture real layer timing (predicted-event timestamp to native
   expert admission), source IO + 13.5 MiB Vulkan upload p99, under
   a memory-safe full-model configuration; otherwise deadlines are
   assumptions.
2. Compare true avoided *misses per token* with/without prefetch,
   include same-token native logits and exact output identity.
3. Test representative independent real prompts rather than the
   current synthetic-ID held-out route sequence.
4. Add multi-session ownership/eviction/cancellation and provider
   admission; disabled pool and speculative IO remain isolated.
5. Existing Joyner full-GLM memory-pressure NO-GO remains active;
   use higher-RAM node or free sufficient memory before full inference.

**Production: HOLD.** No GLM runtime source changes, serving restart,
KV/RoPE modification, speculative token acceptance, or model writes.
The two Joyner production endpoints remained HTTP 200.
