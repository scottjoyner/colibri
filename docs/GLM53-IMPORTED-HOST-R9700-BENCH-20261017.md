# GLM-5.3 real expert uploads — imported host memory on Joyner

**Hardware:** Radeon AI PRO R9700 (PCI `1002:7551`, RADV GFX1201); Samsung 9100 PRO Gen5 SSD.
**Test date:** October 17, 2026, around 02:10–02:20 EDT.
**Source:** Fully verified 62-shard Colibri GLM-5.3-Flash checkpoint, revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`. Reads target 4 MiB aligned regions of `model-00002-of-00062.safetensors`. No model files were changed.

## Why this matters

The previous double-buffer pipeline performed an `O_DIRECT` read into ordinary RAM, then copied that RAM into a Vulkan-mapped staging buffer before submitting the GPU transfer.

A naive direct `O_DIRECT` read into Vulkan-allocated mapped memory **failed with Linux EFAULT (errno 14)**. The new experiment instead allocates ordinary aligned host RAM, imports it into Vulkan using `VK_EXT_external_memory_host`, and uses that **one allocation** for the direct SSD read and Vulkan transfer source. It removes an explicit CPU copy but does **not** eliminate the SSD→RAM→GPU hardware transfers.

The R9700 driver was confirmed to support external host memory, accepting a 4 MiB allocation with 4096-byte alignment, coherent imported memory type 5, and matching transfer-buffer memory bits.

## Correctness before performance

With 1 GiB process RAM hard limit, swap disabled and low priority, two dedicated experiments returned each GPU upload to host and checked all bytes against the read-in staging allocation:
- 128 out of 128 transfers matched.
- 384 out of 384 transfers matched.
- **512 verified transfers = 2,147,483,648 verified bytes, zero mismatches.**

The verification mode performs an extra GPU→host transfer **for every upload** and was not used for throughput comparisons. Throughput mode retains the final-buffer readback check. The underlying model shards had already been SHA-256 verified prior to testing.

## Paired short-trial throughput (128 distinct 4 MiB expert regions per trial)

| Trial | Existing double buffer | Imported host-memory buffer | Increase |
|---|---:|---:|---:|
| 1 | 4.727 GB/s | 7.129 GB/s | +50.81% |
| 2 | 4.740 GB/s | 8.228 GB/s | +73.59% |
| 3 | 4.737 GB/s | 7.896 GB/s | +66.69% |
| **Median** | **4.737 GB/s** | **7.896 GB/s** | **+66.69%** |

Test order alternated imported/baseline and baseline/imported across the three trials using the **same region offsets within each pair**.

## Longer paired throughput (384 distinct regions, 1.611 GB per run)

| Trial | Existing double buffer | Imported host-memory buffer | Increase |
|---|---:|---:|---:|
| 1 | 4.902 GB/s | 7.887 GB/s | +60.89% |
| 2 | 5.047 GB/s | 8.179 GB/s | +62.06% |
| **Median of the two trials** | **4.974 GB/s** | **8.033 GB/s** | **+61.48%** |

The longer tests ran baseline/imported then imported/baseline. The 384 offsets are unique and preserved in the benchmark receipts.

## Reproducible files

- `bench_vulkan_imported_host.c`: hardware-pinned, two-buffer, native `O_DIRECT`→imported-host→Vulkan GPU transfer. Rejects incompatible GPU IDs, missing extension/coherent memory type, unaligned imports, short reads, incorrect verification, and allocation failures.
- `probe_vulkan_import_memory.c`: reads GPU capabilities and verifies the imported host buffer can bind to a Vulkan transfer source.
- `run_imported_host_pairs.sh` and `run_imported_long_pairs.sh`: fixed-offset paired runs with production endpoint health checks.
- `hardware-20261017/imported-host/`: all captured raw JSON, the direct-mapped `EFAULT` negative test receipt, source offsets, and `summary.json` containing SHA-256 digests and comparison statistics.

## Scope and remaining gates

**Not a GLM inference-speed measurement.** The live GLM decoding smoke remains around 0.61 tokens/sec; reducing standalone transfer time may help but cannot establish token latency gains without integrating native expert-cache decisions, concurrent request ownership, tier eviction and GPU completion fencing. The improvements are from isolated short tests on one shard; `O_DIRECT` avoids the OS page cache but not device/controller caches. Under active inference, tail latency and GPU resource contention may change.

**Production HOLD:** no changes to AssistX, model-serving ports, live cache policy, glibc/libvulkan installation or upstream Colibri. The optimized transfer remains an isolated experimental fork harness. All live Joyner model endpoints returned HTTP 200 at final check, and no orphaned staging files remained.

Next engineering gate: add the new buffer type behind a disabled feature flag in the fork's Vulkan expert-tier component, track source revision and per-session cache generation, and prove exact token/logit identity plus real inference tokens/sec against the unchanged model. No live prefetch should occur until those gates pass.
