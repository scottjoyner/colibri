# Joyner Gen5 expert I/O and Qwen3.8-Flash-Next offload evaluation

**Date:** October 17, 2026, 09:25–09:30 EDT. **Host:** Joyner R9700 + Samsung 9100 PRO Gen5.
**Fork:** `scottjoyner/colibri`. **Status:** read-only SSD benchmarks complete, model deployment NO-GO.

## Verified physical SSD reads: do not assume queue depth 8 is faster

Three rounds on the exact 62 verified, source-revision-pinned GLM-5.3-Flash int4 checkpoint,
256 **unique** 4 MiB expert tensor regions per round, with the **same offsets** for each
queue-depth comparison in that round. Reads performed via Linux `O_DIRECT` to ordinary
host RAM, and the order of QD modes changed across rounds.

| Simultaneous read workers | Median O_DIRECT GB/s | Median per-read p99 latency |
|---|---:|---:|
| 1 | 8.307 | **0.733 ms** |
| **2** | **11.319** | **1.056 ms** |
| 4 | 11.324 | 2.641 ms |
| 8 | 11.267 | 4.303 ms |

Total requested *physical* read volume **12.90 GB**. Per-mode whole-block-device
sector counters increased by about 1.075 GB per 256-region round, consistent with
the expected direct I/O volume, although these counters include other host workloads.
3,072 sampled 4MiB transfers (12 experiments × 256, 12.9GB) were issued and validated;
their raw JSON and sample-set SHA256 hashes are in
`benchmarks/joyner-glm53-20261016/hardware-20261017/gen5-paired/`.
The small initial 64-region control was also recorded independently.

**Conclusion:** QD2 is the current best operating point for a layer-critical expert
read under constrained tail latency: virtually the same total throughput as QD4,
with dramatically less p99 read time. QD4 can be a bulk-fill configuration if a
router deadline is not imminent. Blindly raising to QD8 does not help on this
workload. Prefer reducing the number of unique reads (GPU/RAM cache hits and
causal prefetch) before attempting larger submission batches.
`O_DIRECT` bypasses Linux page cache but **not NVMe controller caching**.
These storage-only results do not prove end-to-end tokens/sec.

## Updated upstream Colibri source

Read-only fetched `JustVugg/colibri origin/dev@beb7cfc4` (Oct 17, 2026).
New updates since the previous fetched `22ef2bae` include
DeepSeek V4 prewarm buffer-free correctness, Qwen3.6 GGUF final-norm checks,
and Vulkan RSS-accounting improvements. None justify silently merging all of
upstream's large Vulkan/GLM source changes into our pinned fork.

Already selective-ported upstream `cfab4f65` to our draft PR #7:
`COLI_VK_TIER_FILL=1` fills unused VRAM expert tier slots, while default
retains prior history-only behavior. Vulkan tests pass, but the full GLM
canary on Joyner hit severe host memory PSI **before the opt-in case started**.
Keep it disabled until sufficient host headroom.

The reusable host memory importer is now in draft PR #8:
`COLI_VK_UP_POOL=1`, 12 slots, 64MiB hard allocation cap, existing
fallback. The physical R9700 small fixture passed with 4.16% lower median
wall-time (eight alternating trials), but no full model speedup is established.

## Qwen3.8-Flash-Next: important architecture split

**Do not conflate GLM-5.3, Qwen3.8-27B and Qwen3.8-Flash-Next.** The
official Qwen3.8-Flash-Next-FP8 native safetensor model is ~185.5GB on disk,
including ~120.8GB routed experts, and has 125B main parameters plus a
51B hashed n-gram table. Upstream `c/qwen38.c` streams **16 FP8 n-gram rows
of 160 bytes each per token** and never loads all of the PLE table into host
RAM. That is a 2,560-byte payload per token, though random storage pages
may require more physical bytes. A future optimization would page-coalesce
rows when hashes land on nearby pages, use a small explicitly capped row
cache, and compare batched *random-read IOPS and p99* (not just GB/s).
No PLE cache implementation is included here without the pinned FP8 model.

Upstream's official-FP8 engine has ~**9.2GiB fixed resident dense weights**,
scales, recurrent, KV and workspace overhead: its own documentation measures
**12.9GiB peak RSS at a 16-slot expert cache** and states that 16GB machines
are below the comfortable floor. Joyner has ~14GiB total host RAM and other
live services, making this official FP8 engine an unsafe full-model
benchmark candidate on Joyner now. The Gen5 SSD has ~666GB free,
but disk capacity is not the same as sufficient host-memory headroom.

A **61GB GGUF** of Qwen3.8-Flash-Next already exists on x1-370 at
`~/.lmstudio/models/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF/qwen38-keep1-Q3KXL.gguf`.
It is **not** the official FP8 safetensor layout expected by Colibri's
native `qwen38` engine; copying it to Joyner cannot by itself enable
that engine's PLE/native-FP8 pipeline. The x1-370 root filesystem has
only ~96GB free and cannot currently receive another 185GB full official
FP8 checkpoint without a storage plan. **No large downloads or model
copies were started.**

## Speculative decoding and RoPE/YaRN constraints

Qwen3.8-Flash-Next includes a checkpoint-backed built-in MTP drafting
head. Upstream Colibri exposes `Q38_MTP=0/1`, `Q38_MTP_CAP` and
`Q38_MTP_DRAFTS` with native acceptance/rejection and recurrent-state
rollback. That is the first model-specific speculative-decoding path to
benchmark *after* native weights and adequate RAM are available.

DFlash2's published Qwen3.8 checkpoint currently targets **Qwen3.8-27B**,
not verified Qwen3.8-Flash-Next nor GLM-5.3; do not interpose it into
their native token path. Predictive **expert prefetch** may instead safely
reorder/overlap storage reads so long as final native routing, logits
and token IDs are unchanged.

Qwen's official model card states **native 262,144-token context** and
provides mRoPE/YaRN scaling for context longer than that, with the
position sections `[11,11,10]`, `rope_theta=10000000`, and recommended
factors appropriate to requested length. These parameters **alter
attention semantics**; static YaRN can degrade shorter prompts. Current
Colibri `qwen38` enforces native max position; no beyond-native YaRN
compatibility was verified in this fork. Keep native RoPE settings pinned
and do not interpret YaRN as an SSD speedup.

## Next bounded integration gates

1. Keep the **real GLM** execution tests on HOLD until Joyner host RAM
   can avoid memory PSI; maintain separate 6GB canary accounting.
2. Introduce a capped *two-reader expert prefetch* experiment behind a
   disabled-by-default flag and capture hit/miss/accuracy, physical
   NVMe sector deltas and native-logit equality. Explicit cancellation,
   pinned source revision, in-flight ownership and group lifetime.
3. Benchmark Qwen3.8 Flash Next official FP8 on a node with both adequate
   RAM and storage, without first copying its 61GB GGUF to Joyner
   under the mistaken assumption that it meets that requirement.
4. Separate MTP acceptance/rollback tests from predictive expert
   IO scheduling; only test mRoPE/YaRN past native context with golden
   short- and long-sequence attention/control fixtures.

References: [Upstream Colibri](https://github.com/JustVugg/colibri),
[Colibri qwen38.md](https://github.com/JustVugg/colibri/blob/dev/docs/qwen38.md),
[Official Qwen3.8 Flash Next FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8),
[DFlash supported models](https://github.com/z-lab/dflash).
