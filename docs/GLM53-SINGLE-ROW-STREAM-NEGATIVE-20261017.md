# Colibri GLM-5.3: single-row GPU expert streaming — NO-GO

October 17, 2026, Joyner R9700 32GiB + Ryzen 7600X + Gen5 NVMe.

## Question and design

Can we move every cold MoE expert to the R9700 GPU during single-token
decoding instead of decoding its corresponding expert on the CPU?

Backend Vulkan already has a GEMV route for <16 rows, but the existing
cold-expert stage is only entered for large prompt batches (S>=16).
This branch adds a GLM-only, default-off environment switch
COLI_VK_TIER_SINGLE_ROW_STREAM=1 to enter bounded cold-expert staging when
S==1. Existing bounded host staging, GPU uploads and CPU fallback remain.
Default serving, non-GLM engines, and ordinary GLM benchmarks are unchanged.

Experimental binaries and runtime flags are restricted by the exclusive
Joyner benchmark runner to 4GiB expert tier, promotion rate 32,
text-only mode, host-import disabled, 6GiB memory cgroup max, zero swap,
pressure monitoring, source UUID pin, and CPU/GPU/SSD competitor checks.

## Important rejected preliminary experiment

The first attempt (round c011d5e53a21) reported ~0.749 versus 0.798 tok/s,
BUT the runner omitted COLI_VK_TIER_SINGLE_ROW_STREAM from the
systemd-run environment allowlist. The engine never activated the feature.
Those results are **INVALID AS A STREAMING TEST** and are not a speedup
claim. We fixed the allowlist and added a mandatory engine activation
log check that invalidates a trial if the feature was not enabled.

## Corrected real-checkpoint A/B

Round d11eec011d52, exact same binary SHA-256
b2cf18d0b98da29f868818ad52bd84dd2d33c70d0448edb201863ede250c74a0.
Both tests returned exactly greedy 16 11 21 13 17 11 21 13,
had no detected competitors or watchdog violation, and exited successfully.
The second trial logged the actual engine activation flag.

| Metric | Fair-share control | Single-row stream |
| --- | ---: | ---: |
| Throughput | 0.750 tokens/s | **0.273 tokens/s** |
| Eight-token decode wall time | 10.7 sec | 29.3 sec |
| GPU-served experts | 565/3696 | 2688/3696 |
| Expert reads | 2,961 | 3,424 |
| Expert bytes | 39.04 GiB | 45.14 GiB |
| Host RSS | 3.38 GiB | 3.38 GiB |
| Lowest host MemAvailable | 9,344,932 KiB | 8,903,604 KiB |

**Result: -63.6% throughput**. More GPU activity is not sufficient.
The candidate streamed 2,580 cold experts (34.01 GiB) in 336
single-row stages. Transfer pipeline measured only about 1.89 GB/s,
with 181.7ms aggregate GPU waits versus ~1.9ms in the control.
Streaming also displaced resident experts (96 instead of 224 resident),
and increased expert reads. This implementation trades valuable GPU cache
residency for expensive single-use uploads.

## Decision

**Reject unconditional single-row GPU streaming for production**.
Keep the feature default-off. Do not auto-enable, merge into an admitted
fleet service, or claim the new binary is ready for 24/7 use. Both canary
services on 18320 and 18321 returned HTTP 200 after the test;
exclusive controller verified no orphan benchmark scopes or recovery timer.

The next experiment should test **selective, causally precomputed,
high-confidence expert prefetch into otherwise idle device capacity**,
with explicit cache-residency preservation and a bounded wasted-read
budget; or seek a more compact expert layout. A 72-hour service soak,
reference-logit parity, fault injection, admission fencing and rollback
are still mandatory before any production promotion.

Raw receipts:
- benchmarks/joyner-glm53-20261016/exclusive4fair-d11eec011d52.json
- benchmarks/joyner-glm53-20261016/exclusive4stream-d11eec011d52.json
