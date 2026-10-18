# Joyner GLM-5.3 final-fleet runtime: real-checkpoint memory and fair GPU tier

**2026-10-17, experimental stack above [PR #17](https://github.com/scottjoyner/colibri/pull/17).**
**Production: HOLD; no new GLM listener, admission, model registry entry, or
running endpoint was changed.** This is an isolated CPU/Vulkan optimization,
not a fleet release certification.

## The critical finding: there was enough GPU capacity, but not host RAM

The successful earlier historical 62-shard GLM reference decoded eight synthetic
tokens at **0.630 tok/s** using `GLM53_BOUNDED_DENSE=1`,
`COLI_VK_DENSE_HOST=1`, a 2GiB GPU expert tier, and
`COLI_VK_UP_IMPORT=0`. That run reported **5.56GiB RSS**.
A separate import-enabled test produced severe node-wide PSI and was stopped.
The earlier result is a valid *synthetic full-model execution*, not proof of
safe continuous serving or natural-language generation.

Our new strict watchdog (900,000 KiB minimum node MemAvailable on two samples)
stopped BOTH new full-vision tests before any completed decode, despite a
6GiB process RAM limit and disabled unit swap:

| Starting configuration | Minimum observed MemAvailable | Result |
|---|---:|---|
| Dense host copy kept, 2GiB tier/16-rate | 759,200 KiB | **Safety stop** |
| Dense host copy dropped, 2GiB tier/16-rate | 623,644 KiB | **Safety stop** |

In the second run, the engine correctly dropped **4.29GiB of redundant host
dense copies** but still reported **4.89GiB** of RSS after dense placement.
The remaining ~2GiB of vision-tower f32 allocations were unnecessary for
our text-only prompt. A capacity-only change was not safe at this point.

## New native GLM capability: GLM53_TEXT_ONLY=1

**Explicitly opt-in and OFF by default.** In this mode `vision_load`
does not allocate vision-tower weights. The model declares
`CAPS vision=0` in the serving protocol; a received IMAGE frame that
would be processed with no vision tower yields BAD_REQUEST. CLI
`--patches` is rejected *before model loading*. Text prompts containing
any of the six configured image/video special token IDs are rejected
rather than silently falling back to their text embedding.

The standard image-capable mode is unchanged, and all 62 checkpoint shard
files remain read-only on the Gen5 SSD. This is deliberately a
**capability split**: only the text-serving instance may omit the vision
tower; a later multimodal fleet instance must keep the original behavior.

Physical Joyner tiny-fixture checks: full model-feature mode, text-only mode
and text-only + fair promotion produced **identical teacher-forcing and greedy
token outputs** using the R9700. All six configured visual marker IDs
were refused in text-only mode. Pure fail-closed C visual-policy tests pass
under UBSan.

A subsequent full-checkpoint text-only run completed 8 native tokens with
RSS **3.38GiB**, minimum node MemAvailable **2,562,988KiB** at the 2GiB tier,
with no OOM or unit swap. Compared with the aborted full-vision dense-host-
drop run, the reported post-placement RSS changed from ~4.89 to ~2.79GiB:
approximately **2.10GiB less host residency**. This supports the memory
benefit, though the two runs had different success outcomes and the
estimated saved bytes are not a direct OS allocation trace.

## Fair-share Vulkan expert promotions (opt-in)

New `COLI_VK_TIER_FAIR_LAYERS=1`, GLM-only, default OFF. Instead of
allowing the earliest processed sparse layers to spend an entire
`COLI_VK_TIER_RATE` token budget, compute a bounded *per-layer* share.
With 42 routed layers and a budget of 32 promotions/forward, assign
32 layers one opportunity, rotate the chosen layers every forward.
No layer may exceed its current quota; the existing **global**
budget, async staging queue, eviction, CPU fallback, expert integrity
and GPU completion fences remain authoritative.

The fairness function never looks ahead to future routes, never writes
weights, and is NOT a speculative expert predictor. The policy has
exactly the same default behavior when the flag is absent. Its
per-forward counters reset on each new forward; the policy is bounded
by existing worker and queue capacities and does not add a new
GPU upload buffer.

Unit tests under UBSan confirm that allocations exactly sum to the
unchanged global cap for budgets 0..256 across 20 forward epochs, each
layer differs by at most one opportunity per forward, and every layer
receives equal opportunities over the 42-epoch cycle.
Tiny native Joyner Vulkan flag-off/flag-on generated the same tokens.

## Full-checkpoint physical results (frozen 11,21,13 -> eight greedy tokens)

All completed runs generated exactly:
`16 11 21 13 17 11 21 13`.

Common controls: exact checkpoint, 45/45 dense chain layers on R9700,
`GLM53_BITS=4`, `GLM53_BOUNDED_DENSE=1`,
`COLI_VK_DENSE_HOST=0`, `GLM53_TEXT_ONLY=1`,
`COLI_VK_STAGED=1`, **`COLI_VK_UP_IMPORT=0`**,
`GLM53_COALESCED_EXPERT_IO=vector`,
no warmfill or trace persistence, 2 OpenMP threads,
6GiB cgroup max, MemoryHigh=5800M, MemorySwapMax=0,
low CPU/IO weight, and a pressure/health watchdog.
The default CPU cache is one expert slot per each of 42 sparse layers.

| Native configuration | Decoding | GPU routed | SSD loads | GPU uploads | Engine RSS |
|---|---:|---:|---:|---:|---:|
| 2GiB/16, fair OFF | 0.358 tok/s | 231/3696 | 3299 | 137 | 3.38GiB |
| 4GiB/32, fair OFF | 0.380 tok/s | 418/3696 | 3114 | 326 | 3.38GiB |
| 8GiB/32, fair OFF | 0.363 tok/s | 296/3696 | 3233 | 352 | 3.38GiB |
| **4GiB/32, same-binary fair OFF control** | **0.378 tok/s** | **418/3696** | **3114** | **326** | **3.37GiB** |
| **4GiB/32, same-binary fair ON** | **0.398 tok/s** | **565/3696** | **2961** | **270** | **3.37GiB** |
| 8GiB/32, fair ON | 0.399 tok/s | 579/3696 | 2947 | 352 | 3.38GiB |
| 4GiB/32, fair ON, CPU cache 2 slots/layer | 0.375 tok/s | 565/3696 | 2944 | 270 | 3.92GiB |

**Pivotal comparison:** With the **same binary and all other settings
unchanged**, fair 4GiB/32 achieved **+5.29% observed decode tok/s**,
**147 additional GPU-served expert assignments**, **153 fewer SSD
expert loads** and **56 fewer GPU uploads**. These are two *single
runs*, not a sufficiently repeated or randomized distribution. Do
not call the speed increase statistically established.

**Important negative controls:**
- 8GiB of tier capacity on its own was worse than 4GiB.
- 8GiB + fairness produced essentially identical speed to 4GiB +
  fairness, despite many more uploads and more VRAM usage.
- Two CPU cache slots saved just 17 SSD loads and **slowed** decode
  while consuming an additional ~0.55GiB of host RAM.
- Our best new fair 4GiB text-only time is **still below** the earlier
  0.630 tok/s full-vision synthetic reference, which ran with
  different host-memory and runtime conditions. Do not compare
  that historical number as if it were a paired A/B.

### Recommended next isolated candidate (NOT production configuration)

```text
GLM53_TEXT_ONLY=1
GLM53_BOUNDED_DENSE=1
GLM53_DENSE_SPILL_DIR=/srv/joyner-gen5/colibri-fork-dense-stage
GLM53_BITS=4
GLM53_EXPERT_GB=0.05
GLM53_COALESCED_EXPERT_IO=vector
COLI_VULKAN=1
COLI_VK_DEV=1
COLI_VK_CHAIN=1
COLI_VK_DENSE_HOST=0
COLI_VK_TIER_GB=4
COLI_VK_TIER_RATE=32
COLI_VK_TIER_FAIR_LAYERS=1
COLI_VK_TIER_REPEAT_ADMIT=0
COLI_VK_TIER_WARM=0
COLI_VK_STAGED=1
COLI_VK_UP_IMPORT=0
COLI_VK_UP_POOL=0
USAGE_SAVE=0
KVSAVE=0
OMP_NUM_THREADS=2
```

Do not run this as an unattended production service. All native GPU
tests were isolated transient scopes; their logs remain privately on
Joyner under
`/home/joyner/glm53-native-runtime-shadow-20261017/physical_tier_ab_20261017`.
The GitHub PR contains aggregate receipts, not sensitive trace rows or
checkpoint tensor payloads.

## Safe reproduction and promotion requirements

`benchmarks/joyner-glm53-20261016/run_joyner_native_tier_ab.py`
verifies Joyner's R9700 identity and SSD staging UUID before launching
a transient non-root systemd scope. It checks the pre-existing model
endpoints, RAM and VRAM snapshots, node memory PSI each second, and
imposes a 6GiB hard resident budget, zero unit swap, 100-second model
timeout, 900,000KiB host-memory floor and 4GiB GPU-free floor. Two
consecutive pressure violations stop the isolated scope. A successful
run requires **the exact expected token IDs and count**, complete
GPU routing telemetry and no watchdog stop. The controller closes
the transient scope on its own exception as well as the normal path.

The first two variants physically triggered the pressure watchdog
and were stopped with both existing production ports staying HTTP 200.
All seven text-only physical variants completed. No GLM service was
registered, deployed or exposed through the fleet API; the model still
has a **production NO-GO**.

Before promotion: reverse-order and multi-seed/multi-prompt A/B
repetitions; natural-language reference logits and quality checks;
p50/p99 layer latency, startup and cancellation tests, long-context
PSI/VRAM thresholds, CPU-downgrade behavior after GPU loss,
session-fenced admission, multiuser quotas, and model source
cryptographic validation. The new text-only variant must never
advertise image support.

**Rollback:** omit `GLM53_TEXT_ONLY` and
`COLI_VK_TIER_FAIR_LAYERS`; preserve normal Colibri behavior. In
any low-RAM Vulkan staged experiment, explicitly keep
`COLI_VK_UP_IMPORT=0`.
