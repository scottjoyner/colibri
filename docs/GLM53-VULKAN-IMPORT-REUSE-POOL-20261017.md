# Reusable Vulkan imported-host images: fork-only prototype

**Date:** October 17, 2026. **Status:** opt-in experimental code, no full-model promotion.

The standalone R9700 experiment with two persistent imported buffers reached
~8 GB/s for 4 MiB expert transfers. In contrast, on Joyner with 14 GiB host RAM,
a full GLM-5.3 run using per-tensor `COLI_VK_UP_IMPORT=1` failed to complete
and exhibited node-wide memory pressure. It was stopped; the known safe fallback
is `COLI_VK_UP_IMPORT=0` with forced staging.

This patch augments Colibri's existing `up_img_alloc`/commit path:

- **Off by default.** Only `COLI_VK_UP_POOL=1` uses the pool, and only
  if staged Vulkan uploads and `VK_EXT_external_memory_host` are available.
- **64 MiB absolute import reservation limit, 12 retained allocations.**
  Primary Vulkan device only. No import-cache extension to a second GPU;
  that device uses the existing path.
- Host images are ordinary page-aligned coherent RAM, imported as Vulkan
  transfer-source buffers once and reused. Before each reuse, source bytes
  are zero-filled. Source ownership stays exclusive until the same
  synchronous `coli_vk_tensor_commit` completes GPU fences.
- `up_import` recognizes a checked, live image inside the pool and reuses
  its VkBuffer without creating/binding a fresh Vulkan object.
  Other pointers follow the original importer, with fallback to staging
  when the driver refuses a pointer. The fault-injection path remains.
- Failed allocations, invalid memory types, resource cap exhaustion or
  missing device capability leave the original staged importer available.
  No routing, source shard, RoPE positions, expert eviction or
  quantization semantics change.
- The pool is drained before Vulkan device destruction. No pointers are
  retained after the owning process terminates.

## Completed x1-370 evidence

`VK=1 tests/test_vk_tier` passes with
`COLI_VK_STAGED=1 COLI_VK_UP_IMPORT=1 COLI_VK_UP_POOL=0` and with pool `=1`.
That includes grouped/streaming expert compute, warm-fill, DeepSeek V4
quantization, and fault-path regression checks. A pooled fixture printed:

```text
[VK] reusable host imports: created 12, reused 4593, fallback 45, peak 0.63 MiB (cap 64 MiB)
PASS
```

The counters are from this synthetic fixture, not GLM's real 13.5 MiB
expert tensors. The import pool's retained pages might worsen RAM
contention in a full model despite reducing allocation churn.
**No GLM tokens/sec improvement is claimed.**

## Required R9700 acceptance before any production path

1. Run the existing tiny Vulkan test on Joyner with pool disabled/enabled,
   checking exact numerical parity and fault behavior; verify actual
   imported source count and bounded retained memory.
2. Before a new full GLM canary, first stabilize host RAM headroom:
   the first cold-fill canary was stopped for memory PSI, independent
   of whether import pooling was enabled.
3. In the first bounded full-model test, abort on memory PSI and low
   `MemAvailable`; compare native logits/greedy outputs,
   zero/nonzero staging imports, each upload's target consistency,
   true disk reads, and end-to-end token/sec.
4. Reproduce across workload-specific prompt traces; no YaRN/RoPE
   or DFlash2 token-draft changes are part of this patch.

Rollback: omit `COLI_VK_UP_POOL=1`, or set it to `0`.
Production/AssistX keep their existing binaries and settings.
