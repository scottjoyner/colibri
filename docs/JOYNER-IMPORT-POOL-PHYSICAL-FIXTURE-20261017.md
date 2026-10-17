# Joyner R9700 — reusable imported-host buffer A/B receipts

Date: October 17, 2026, approx 03:17–03:19 EDT.

The production GLM checkpoint was **not loaded** in this test.
Source code: fork `experiment/joyner-vulkan-import-reuse-pool-20261017`;
R9700 PCI `1002:7551`, RADV GFX1201; Vulkan/Colibri expert tier fixture.

## What changed

With `COLI_VK_UP_POOL=1` only, a maximum of 12 aligned
`VK_EXT_external_memory_host` imports can remain allocated for reuse.
Their aggregate retained host allocation is capped at 64 MiB.
Buffers remain exclusive until the GPU fence is complete. The original
dynamic importer and staged memcpy remain the fallback, and this
feature is off by default.

## Physical A/B

One warm-up/pair run and **eight additional alternating fixture trials**
used identical compiled source, model fixtures, Vulkan config and
two OpenAI-style production health checks per iteration. The benchmark
used an isolated systemd unit with `MemoryMax=1G`,
`MemorySwapMax=0`, low CPU/IO priority.

| Metric | Existing import | Reusable pool |
|---|---:|---:|
| Runs in timed series | 4 | 4 |
| Median full fixture wall time | **2.644 s** | **2.534 s** |
| Change | — | **−4.16% elapsed time** |
| Correctness suite | PASS | PASS |
| Pool import lifecycle | Disabled | 12 created; 4,593 reused; 45 fallback |
| Peak retained imported RAM in small fixture | — | **0.63 MiB** |

One additional pool-backed Vulkan device recreation later reports
24 cumulative imports / 4,605 reuses; counters accumulate across device
initializations within one fixture process. All control and pooled
runs pass the numerical expert-route tests, warm-fill test and
negative driver-injection tests.

The JSON and eight full logs are under
`hardware-20261017/pool-fixtures/`.
All fixture-log SHA256 digests are present in `summary.json`.
These are **not** full GLM tokens/sec figures and the time difference
is not evidence of stable production inference gain.

## Why the full-model canary remains NO-GO

A separate isolated Joyner full-checkpoint warm-fill A/B was
attempted from commit `8caf59e0`. The very first history-only
`COLI_VK_UP_IMPORT=0` run suffered severe memory PSI and never
finished a valid decode; we stopped the service. Its captured
full.avg10 reached **25.94%**, even before an opt-in cold-fill run
started. The 14 GiB host cannot accommodate the full-model process
plus existing services with reliable latency at the current
6 GiB per-process allowance. These logs are preserved under
`hardware-20261017/warmfill-aborted/`; no speedup is claimed.

The benchmark wrapper and host controller had no ability to change
live inference, and both existing endpoints remained HTTP 200.
On cleanup, Joyner recovered to approximately 6.2 GiB available,
with near-zero short-window PSI.

**Next GO criteria:** free enough host RAM or relocate the full-model
benchmark to a machine with materially more memory; then compare
same-prompt tokens/logits, p99 expert-read deadlines, GPU cache hit
rate and memory PSI across the flag combinations. Keep the warm-fill
and pooled importer disabled until that is reproducible.
