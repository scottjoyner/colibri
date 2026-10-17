# GLM-5.3 real-weight read → Vulkan transfer benchmark (Joyner R9700)

**Date:** 2026-10-17 01:21–01:30 EDT  
**Fork:** `scottjoyner/colibri`, stacked on draft predictor PR #3.  
**Source:** `zai-org/GLM-5.3-Flash@46a050bec13002ad29ee795c16e61cf4fa3c4a9c`. All 62 converted shards and 194.73 GB were independently SHA-256 verified before this experiment. Sample shard: `model-00002-of-00062.safetensors`, Samsung 9100 PRO Gen5 SSD mounted at `/srv/joyner-gen5`.

## Physical devices and safeguards

- Discrete GPU: AMD Radeon AI PRO R9700 / RADV GFX1201; Vulkan physical PCI `1002:7551` (driver reports `AMD Radeon Graphics`, not literal `R9700`).
- GPU staging: host-visible coherent Vulkan buffers; destination: device-local buffer. Test validates the **last** 4 MiB GPU upload against its original bytes through an independent GPU-to-host copy. **It does not individually verify all 128 transfers.**
- Read path: Linux `O_DIRECT`, 4 MiB aligned reads from 128 real converted expert tensor regions; no model shard writes, no synthetic byte payload.
- Process units: max 1 GiB RAM, `MemorySwapMax=0`, low CPU/IO weights and runtime bound. Live model services on 18320 and 18321 continued reporting HTTP 200. No AssistX/model-server changes.
- Vulkan numbers are end-to-end host-submission + fence-wait latencies, not a pure PCIe microarchitectural bandwidth counter. O_DIRECT bypasses Linux page cache but **not internal SSD controller caching**.

## Isolated expert O_DIRECT concurrency sweep

256 expert tensors per configuration, each ~4 MiB (~1.075 GB):

| Concurrent readers | GB/s | p50 latency | p99 latency |
|---|---:|---:|---:|
| 1 | 8.27 | 0.488 ms | 0.738 ms |
| 2 | 11.22 | 0.698 ms | 1.662 ms |
| 4 | **11.54** | 1.390 ms | 2.635 ms |
| 8 | 11.23 | 2.856 ms | 4.494 ms |

The throughput optimum differs from the latency optimum. For a layer-specific prefetch deadline, a single reader's lower tail can outweigh peak aggregate GB/s. Keep a bounded queue and test I/O QoS under inference load separately.

## Actual SSD→CPU→GPU transfer, sequential vs double buffered

The **single-buffer** benchmark read an expert region from NVMe into aligned host RAM, copied it into a Vulkan host-visible staging buffer, submitted a device-local 4 MiB GPU transfer, and waited for the fence. The **double-buffer** benchmark overlapped its next read/staging copy with the preceding Vulkan submission using two staging slots and a dedicated uploader thread.

One initial 128-region test: serial **3.673 GB/s**, double buffer **4.450 GB/s**, both byte-readback checked.

Three further paired trials on **384 distinct real expert regions**, 128 per trial (test order pipeline/serial, serial/pipeline, pipeline/serial):

| Trial | Serial GB/s | Double-buffer GB/s | Relative gain |
|---|---:|---:|---:|
| 1 | 3.951 | **4.756** | 20.37% |
| 2 | 3.954 | **4.766** | 20.54% |
| 3 | 3.954 | **4.664** | 17.96% |
| **Median** | **3.954** | **4.756** | **20.37%** |

Single-buffer transfer component (128 samples): O_DIRECT read **p50 .501 ms, p99 .655 ms**; CPU staging **p50 .304 ms, p99 .344 ms**; GPU submit+fence **p50 .217 ms, p99 .243 ms**. Double buffering improves aggregate wall-time throughput but **worsens p99 read and staging** in some trials. Evaluate whether tail jitter prevents a speculative expert from arriving before native routing.

**No live-cache speedup is established.** The previous full GLM checkpoint inference yielded ~0.61 generated tok/s and heavy expert misses. Multiplying transfer GB/s by those expert byte counts does not predict end-to-end tok/s because CPU work, eviction, synchronous dependencies and cache reuse are not accounted for.

## Reproducibility

C sources: `bench_vulkan_expert_transfer.c` (single-buffer), `bench_vulkan_expert_pipeline.c` (double buffer); sample/paired script: `run_paired_transfer_bench.sh`. The C sources compile with the local Vulkan SDK headers and `libvulkan.so.1` on Joyner. They exclusively select PCI `1002:7551`; fail if the R9700 is not present. All aggregate JSON receipts are in `hardware-20261017/`; the 384 source offsets can be regenerated with the fixed random seed.

**Next GO/NO-GO:** measure PCIe/Vulkan upload and disk/page-cache behavior during real multi-layer inference, add admission deadlines and quota enforcement with occupancy simulation, compare native cache vs Least-Stale, preserve exact-token/logit proofs, and only then trial async speculative transfers in an isolated disabled-by-default model canary. **Production NO-GO.**

