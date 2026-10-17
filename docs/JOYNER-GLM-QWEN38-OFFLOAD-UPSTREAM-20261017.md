# Joyner GLM/Qwen3.8 Flash Next — upstream review and bounded cache rollout

**October 17, 2026.** Fork source: `scottjoyner/colibri`, stacked on
GLM-5.3 imported-host negative runtime A/B. Production inference remains unchanged.

## Upstream fetch and selective port

Read-only fetched `JustVugg/colibri`:
- `main@bf244291` (Oct 6) and `dev@22ef2bae` (Oct 17, local fetch);
  do not merge all `dev`: it changes hundreds of Vulkan, GLM and qwen38 lines.
- Upstream commit `cfab4f654210935ad32d911e2e5cd449e976b4c6`,
  October 14, added cold-cache warm-fill: after history-ranked experts are
  selected, fill remaining GPU capacity round-robin across layers.
  The fork ports just `vkt_plan` and `vkt_warm_report` into GLM-5.3,
  adding a **fork-specific disabled-by-default gate**:
  `COLI_VK_TIER_FILL=1`. Unset/zero retains history-only selection.
  `COLI_VK_TIER_WARM=0` still disables all warm-loading.
- Upstream `dev` has grouped MoE Vulkan/kernel changes, Qwen3.8 chat reuse,
  partial-chain improvements, and a Qwen3.8 MTP decoder. These are
  **not merged** or backported into this safety-sensitive GLM branch.
- The updated warm-fill plan never replaces history-priority slots, selects
  only existing MoE layers, obeys the physical tier limit, never schedules
  duplicate (layer, expert) IDs, and preserves GPU-commit fences.
  The new regression compares disabled/opt-in behavior with deterministic
  toy expert weights.

## Checks

On x1-370, `make -C c VK=1 glm53` and `tests/test_vk_tier`
completed (PASS). The standalone test includes both warm modes and
native grouped/MXFP4 and fault-injection checks; it does not establish
production GLM correctness on Joyner. Do not enable on Joyner without a
controlled single-run A/B with full token/logit identity and memory PSI
abort rules.

## SSD bandwidth: optimize misses, not just sequential GB/s

**Real Joyner measurements** with 256×4 MiB aligned O_DIRECT reads:

| Concurrent reads | Physical read throughput | p99 read latency |
|---|---:|---:|
| 1 | 8.27 GB/s | 0.738 ms |
| 2 | 11.22 GB/s | 1.662 ms |
| 4 | 11.54 GB/s | 2.635 ms |
| 8 | 11.23 GB/s | 4.494 ms |

Peak aggregate QD4 improves bulk reads, but QD1 gives the best
individual-read deadline. GLM's typical expert transfer is ~14.2 MB;
its 42 layers × 8 experts can demand ~4.8 GB per token in a cold run.
On the bounded 3-ID/8-token smoke, 3,299 expert loads and only 231/3,696
routed-expert selections (6.2%) served by the GPU. The limiting question
is **how many unique expert bytes can be avoided**, not just reading
those bytes faster.

The separate imported-host buffer benchmark reached a 7.9–8.0 GB/s
paired median, but GLM's existing short-lived per-tensor Vulkan imports
stalled amid severe memory pressure and were stopped. This fork must
**not** set `COLI_VK_UP_IMPORT=1` on full GLM with Joyner's constrained
memory configuration until bounded reuse/pinning is proven. The known
working fallback is forced-staging with `COLI_VK_UP_IMPORT=0`.

Suggested experimental lane: cap 1–4 concurrent expert reads according
to measured slack before the router dependency; track actual physical
IO per layer, OS page-cache hits separately, p99 read deadlines and
Vulkan upload latency. Distinguish SSD cache versus Linux page cache,
align/checkpoint-pin offsets, and do not expand RAM cache if Memory PSI
becomes nonzero.

## Qwen3.8-Flash-Next, n-gram offload and RoPE

Upstream `c/qwen38.c` and `docs/qwen38.md` already implement
Qwen3.8-Flash-Next native safetensor streaming, expert LRU, and a
checkpoint MTP drafter with exact native verification; this is a
**separate model family** from the GLM-5.3 weights on Joyner.

- Qwen3.8 has 125B regular parameters, roughly 6B active/token, and
  a separate 51B hashed n-gram embedding table. The PLE lookup table is
  a promising direct SSD/CPU-memory-offload target: cache only selected
  rows and preserve n-gram key/row order. This is not a reason to
  duplicate the entire table into GPU RAM.
- Qwen3.8 native context is 262,144 tokens. For longer contexts,
  model-provided YaRN/RoPE settings must be explicit, tested against
  shorter-context behavior, and not silently enabled at startup.
  Causal RoPE/YaRN changes affect attention positions; expert-cache
  transfer optimizations MUST NOT alter logits or positions.
- DFlash2 has an explicitly listed Qwen3.8-27B drafter, **not**
  a verified GLM-5.3 or Qwen3.8-Flash-Next checkpoint. Token drafting
  requires its own accept/reject and state-rollback tests; speculative
  expert *prefetch* only changes data movement and never overrides routing.

References:
- Upstream Colibri: https://github.com/JustVugg/colibri
- Qwen3.8 Flash Next docs: https://github.com/JustVugg/colibri/blob/dev/docs/qwen38.md
- vLLM Flash Next recipe: https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml
- DFlash2 checkpoint list: https://github.com/z-lab/dflash

## Promotion gates still pending

1. Validate opt-in warm fill with a bounded full-model Joyner canary
   without any imported-host runtime objects. Compare native greedy IDs
   and, ideally, exact numerical logits against baseline.
2. Isolate GPU/host overhead from per-tensor import operations and
   measure memory PSI, page pinning and lifetime by batch under a
   small explicit reservation; benchmark reusable source allocations.
3. Simulate limited slots across sessions and eviction policies with
   source pin and cache generation fences. Candidate usefulness per
   byte and p99 layer deadline outrank aggregate GB/s alone.
4. Only then test Qwen3.8 Flash Next with pinned revision and artifact,
   n-gram row streaming and MTP on/off; test native 262k position behavior
   before considering YaRN extension.

No public service, model routing or SSD file transfer was changed.
