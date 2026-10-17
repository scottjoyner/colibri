# GLM-5.3 complete-expert grouped reads — physical Joyner A/B

**Date:** October 17, 2026. **Status:** experimental, read-only, draft.
**Hardware:** Joyner, Samsung 9100 PRO Gen5 NVMe, Radeon AI PRO R9700.
**Model:** pinned `zai-org/GLM-5.3-Flash@46a050bec13002ad29ee795c16e61cf4fa3c4a9c`,
62 locally converted safetensors shards.

## Result and guardrails

Independent header indexing verified 12,384 six-tensor GLM experts:
11,690 fit one contiguous source extent, 693 fit two, and one requires
four. Strict grouping does not bridge different files or discontiguous
tensor gaps. Grouping does **not** omit a projection or quantization scale;
one full expert still contains 14,155,776 logical bytes (13.5 MiB).
All 1/2/4-run example experts and a non-4KiB EOF boundary passed
byte-for-byte checks against the original six tensor payloads.

A **physical A/B/B/A** replay then used the *same* 462 held-out native
routing events from PR #12, the *same* previously frozen prediction
policy and 0.40 validation-fitted second-candidate gate. Runs 01–04
train; run05 calibrates; run06-v2 is untouched during prediction.
Exactly 714 complete experts were physically read in **each pass**.

| Physical complete-expert metric | Six distinct tensor reads | Grouped contiguous extents |
|---|---:|---:|
| Read calls, per 462-event pass | 4,284 | **730 (-82.96%)** |
| O_DIRECT bytes | 10,124,771,328 | **10,110,214,144** |
| Median latency, AB order | 9.200 ms | **8.098 ms** |
| p95 latency, AB order | 10.064 ms | **8.635 ms** |
| p99 latency, AB order | 10.959 ms | **8.894 ms** |
| Median latency, BA order | 9.265 ms | **8.120 ms** |
| p95 latency, BA order | 10.427 ms | **8.729 ms** |
| p99 latency, BA order | 11.042 ms | **9.403 ms** |
| QD2 per-event p99, AB order | 11.118 ms | **9.218 ms** |
| QD2 per-event p99, BA order | 11.140 ms | **9.531 ms** |

*Physical disk-sector deltas matched O_DIRECT bytes in all four runs*.
The 32-event pilot is also archived but is too small to establish
tail-latency improvements. Grouped reads trimmed just 14.56 MB of
4KiB padding (~0.14%) per full pass: the measured improvement comes
primarily from syscall/buffer/alignment reductions, **not** a smaller
13.5 MiB expert payload. The test includes Python mmap allocation
overhead; not yet a native C kernel with fixed buffers.

No server process was restarted. Both production Vulkan inference
endpoints returned HTTP 200 before and after. Host memory PSI stayed
at 0.00 throughout. The physical reads used a dedicated
`systemd-run --user --scope -p MemoryMax=384M -p MemorySwapMax=0`
process with `ionice -c3 nice -n 10`, QD2, pin validation, exact source
descriptor identity and low-memory fail-closed checks.

## Independent size-matched GPU transfer experiments

These experiments read **13.5 MiB aligned windows** from verified GLM
safetensors on a single shard. They intentionally test transfer size;
the aligned window is not claimed to be the exact six-tensor assembled
buffer. They are **not** joined to the routing replay above.

128 shuffled unique offsets from one shard; paired serial/imported and
imported/serial order; pinned R9700 Vulkan physical device
(vendor 1002, device 7551).

| 13.5 MiB per iteration | CPU staging + Vulkan copy | O_DIRECT into imported host allocation + Vulkan copy |
|---|---:|---:|
| Median physical read, AB / BA | 1.221 / 1.218 ms | 1.409 / 1.416 ms |
| Median CPU copy, AB / BA | 1.024 / 1.025 ms | **0 / 0 ms** |
| Median GPU copy + fence, AB / BA | 0.867 / 0.869 ms | 0.813 / 0.800 ms |
| Overall pipeline, AB / BA | 4.241 / 4.246 GB/s | **9.472 / 9.443 GB/s** |

The host-import path uses `VK_EXT_external_memory_host`, a persistent
double buffer, explicit transfer fence, and readback comparison. A
**separate verification run checked 128/128 uploads byte-for-byte**
with `--verify-every`, but the resulting ~0.325 GB/s includes the cost
of synchronously reading every upload back and is **not** a throughput
comparator. The non-verification paired runs checked the final upload
with GPU readback. Direct O_DIRECT into a *regular Vulkan mapped
staging allocation* failed with Linux EFAULT; ordinary Vulkan mapped
memory cannot be substituted for a supported host-import allocation.

**Critical context:** A prior experimental imported-host full-GLM
configuration on this 14 GiB machine triggered memory PSI stalls.
The reusable importer and grouped reads are isolated benchmarks, not
approval to enable an imported expert pool in production.

## Reproduction

Files are under `benchmarks/joyner-glm53-20261016/`:

- `benchmark_coalesced_complete_expert.py`: source pin, six-tensor
  grouping, exact byte checks, per-event QD2, read-only O_DIRECT, sector
  counters, source FD/mtime identity, and fail-closed cgroup checks.
- `bench_vulkan_full_expert_size.c`: serial size-matched GPU benchmark.
- `bench_vulkan_full_expert_size_imported_host.c`: size-matched,
  reusable imported-host GPU benchmark.
- `c/tests/test_glm53_coalesced_expert_io.py`: independent fixture,
  overlap, shard boundary, descriptor and EOF tests.
- `hardware-20261017/coalesced-expert/`: aggregate private-hardware
  receipts, not model weights, raw trace rows or offsets.

On Joyner after copying the benchmark and its PR #12 Python imports
into a private directory:

```bash
python3 benchmark_coalesced_complete_expert.py --verify-only
systemd-run --user --scope -p MemoryMax=384M -p MemorySwapMax=0 \
  ionice -c3 nice -n 10 python3 benchmark_coalesced_complete_expert.py \
    --rows 462 --gated --execute-io --out physical462_gated_abba.json
```

The script **defaults to validation only**; physical ABBA requires
`--execute-io`, cgroup v2 `memory.max<=512MiB` and swap maximum
equal to zero. Source files are never padded, written or changed;
unaligned shard EOFs use a separate read-only buffered tail under 4096
bytes. Tests of extra tens-of-milliseconds latency under foreground
inference should not be attempted without explicit resource admission.

## Remaining release gates

1. Full six-tensor read **directly into a reusable imported host
   allocation**, with correct tensor byte offsets, bounded memory and
   readback parity for the 1/2/4 extent layouts and EOF.
2. Real native GLM router/layer timestamp traces; measure actual
   speculative lookahead slack and overlap with Vulkan upload.
3. Representative natural prompts with exact logits/token parity and
   observable tokens/sec, not merely read-call or copy microbenchmarks.
4. Weighted predictor **recall versus wasted SSD/GPU traffic**; the
   gated PR #12 policy covers only 9.659% of selected expert labels.
5. Durable cache admission, session generation fencing, eviction and
   cancellation tests; no cross-session stale predictions.
6. Host-memory pressure mitigation / larger-RAM placement before full
   GLM inference on Joyner. No full-model promotion from a shadow test.

**Production NO-GO:** existing servers, weights, KV cache, native
router, speculative decode, RoPE and inference deployment unchanged.
