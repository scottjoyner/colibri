# GLM-5.3 direct-to-Vulkan staging experiment

Status: **code-only candidate, not validated on R9700; no live prefetch allowed.**

The earlier physical baseline used a regular CPU buffer plus Vulkan staging and achieved 3.954 GB/s serial vs 4.756 GB/s double-buffered median across three paired trials (+20.37%). The extra CPU memory copy took about 0.304 ms per 4 MiB expert at the median.

This experiment reads aligned expert bytes directly into a Vulkan host-visible, coherent staging buffer, potentially removing the extra CPU copy. Its GPU staging slot stays owned by the producer during the direct read, then by the uploader until the Vulkan completion fence. Mismatched mapping alignment, failed or partial reads, and GPU mismatches fail closed. The optional `--verify-every` independently reads each GPU transfer back and compares all 4 MiB against the source staging buffer; this correctness mode adds work and is not a throughput comparison.

**Completed:** Strict C compile with warnings treated as errors on x1-370, and argument-validation smoke test. **Not completed:** actual R9700 execution and comparative measurements, because remote execution was blocked. No speed improvement is claimed.

Acceptance on Joyner requires its exact PCI ID 1002:7551, same verified 62-shard GLM source and offsets, 1 GiB memory/no-swap scope, per-transfer verification pass, then repeated paired ABBA tests against the unchanged double-buffer binary. Record p50/p95/p99 and observed full-pipeline throughput; distinguish SSD, Linux page cache, PCIe, and GPU events.

Linux direct I/O to driver-mapped Vulkan staging memory may return EINVAL or EFAULT even when aligned; refuse the experiment if this occurs rather than silently falling back to a different transfer path. No data-path changes to production inference or AssistX are included.

A passing isolated benchmark is not evidence of GLM token-generation improvement: real inference, deterministic output parity, tier-capacity behavior and per-session fences are required before any runtime rollout.