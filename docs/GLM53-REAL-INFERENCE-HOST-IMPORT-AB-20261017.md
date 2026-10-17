# Full GLM-5.3 imported-host Vulkan A/B — negative runtime finding

**Date:** October 17, 2026, 02:34–02:38 EDT.
**Checkpoint:** `zai-org/GLM-5.3-Flash@46a050bec13002ad29ee795c16e61cf4fa3c4a9c`, all 62 converted shards verified before experiment.
**Host:** Joyner AMD Radeon AI PRO R9700 (RADV GFX1201), 14 GiB host RAM.
**Code:** unchanged GLM engine from the Colibri low-memory fork (exact same binary between A/B runs).
**Prompts:** synthetic token IDs `11,21,13`, greedy generation of 8 tokens. No private prompts captured.

## Why this A/B was necessary

The isolated imported-host SSD→Vulkan benchmark in draft PR #5 achieved 7.896 GB/s vs 4.737 GB/s for the previous double-buffer transfer pipeline (three paired 128-region trial medians). **However, the existing Colibri Vulkan backend already implements host-memory import** in `up_import`, enabled by default when staged uploads are active. Routed-expert `coli_vk_tensor_commit` can invoke this function.

The previous GLM run did not use staged uploads on Joyner, so the standalone gain did not establish an inference benefit.

## Controlled full-checkpoint comparison

Both runs set `COLI_VK_STAGED=1` to activate the device-local staged-upload path, `COLI_VK_CHAIN=1`, `COLI_VK_TIER_GB=2`, `GLM53_BOUNDED_DENSE=1`, `GLM53_EXPERT_GB=0.05`. An isolated systemd unit capped process RAM at **6 GiB**, disabled unit swap and used low IO/CPU priorities.

| Measurement | `COLI_VK_UP_IMPORT=0` | `COLI_VK_UP_IMPORT=1` |
|---|---:|---:|
| Load | 30.1s | 30.3s |
| Prefill (3 IDs) | 5.1s | 5.0s |
| Decode | **8 tokens in 12.7s** | **No completed response before cancellation** |
| Decode throughput | **0.630 tok/s** | **Unknown; must not claim a token rate** |
| Controller wall | 48s, complete | At least 136s, operator-stopped |
| Reported expert loads (completed only) | 3,299 | Not available |
| Routed experts on GPU (completed only) | 231/3,696 | Not available |
| Staged upload accounting (completed only) | 6,382.1 MiB in 2,147 copies / 990 submits | No completed shutdown report |
| Native greedy token IDs | `16 11 21 13 17 11 21 13` | **Output parity unverified** |

During the imported-host run, Joyner's system-wide Linux memory-pressure monitor reported **full avg60 ≈ 20.59%**; `MemAvailable` fell to roughly **198,828 KiB**. The full checkpoint process was stopped to preserve host availability. The cgroup's swap limit was zero; the unit remained bounded. On stop, node available memory recovered to approximately **5.9 GiB** and recent PSI pressure decreased. Both existing inference services at ports 18320/18321 continued returning HTTP 200.

### Interpretation — important caveats

This is a meaningful **NO-GO**, but it does not prove which import operation causes the stall. Memory pressure is node-level, not specifically attributed to the CUDA/Vulkan import operation; we did **not** finish generation, measure target-output parity, or establish an import-on tokens/sec number. The negative result nevertheless rules out activating the current per-tensor imported-host runtime path under these constrained conditions.

Source inspection shows `up_import` creates and later releases imported Vulkan buffer/device-memory handles for each staged host image. In contrast, the successful standalone experiment **reused two imported buffers** throughout all transfers. The cost of repeated import/unimport, pinning, allocations and page reclaim is now an engineering hypothesis that requires a bounded isolated test. The attempted separate import-churn probe was **not validated** after execution controls blocked its launch; its results are not included.

## Next optimization gates

1. Implement a bounded **reusable import buffer pool** or import-object cache only in a standalone tester first; cap pinned pages and prove fair scheduler behavior under pressure.
2. Measure import/create/free, memory PSI, resident/page-lock memory and latency tails under a 1 GiB bound using unique 4 MiB host allocations (not just repeated GPU copies).
3. Reconcile with existing `VkUp` slot ownership. Never free imported pages before a completion fence. Fail closed on invalid alignment, source revision or cache generation.
4. Compare sequential and alternating real GLM runs **only after host-memory headroom is available**, using actual native logits and token parity and an explicit pressure-based abort criterion. Preserve no-import baseline.
5. Focus separately on the much larger unresolved bottleneck: 3,299 expert loads and only 6.2% native GPU-served routed-expert selections on the completed baseline. A faster upload path alone cannot solve cache misses.

**Production GO/NO-GO: NO-GO.** No model-service restart, no AssistX routing change, no speculative prefetch, no upstream merge. All experiment logs, launch script and machine-readable summary are archived under `benchmarks/joyner-glm53-20261016/hardware-20261017/full-glm-import-ab/`.
