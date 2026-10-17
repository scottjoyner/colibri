# GLM-5.3 causal expert predictor — physical QD2 O_DIRECT replay on Joyner

**October 17, 2026** · Joyner AMD Radeon AI PRO R9700 / Samsung 9100 PRO Gen5 SSD.
**Fork:** `scottjoyner/colibri`, stacked on experimental SSD benchmark PR #9.
**Status:** Shadow-only, no native routing or production model changes.

## Purpose

The previously tested two-reader Gen5 path sustained **11.319 GB/s** on
real GLM expert tensors (p99 1.056 ms for individual 4 MiB reads).
That alone does not tell us whether the predictor reads **useful**
experts. This experiment crosses the previously frozen causal predictor
with **real 4 MiB disk reads from precisely the predicted experts**.

The source checkpoint is `zai-org/GLM-5.3-Flash` pinned to
`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`; all 62 converted shards
had already passed SHA verification. Before any read, the benchmark
checks exact source pin, 62 matching manifests, Gen5 mount identity,
4 KiB alignment, and each mapped up-projection tensor's range.
The model contains **12,384 mapped expert up-projection tensors**.

The training evidence consists only of native routing runs 01–04.
Frozen fusion weights are **previous token 1.0 / cross layer 2.0 /
layer frequency 0.25**. The untouched held-out run06-v2 contributes
462 routing events and full RAM/VRAM bitmaps. For each event, the
current native route **is not consulted** when ranking or selecting
predictions. It is used afterward only to score usefulness. All
per-run trace inputs remain private on the fleet; public receipts
include their SHA256 provenance, not the raw activation-derived data.

## Full held-out results

All policies were given the same **maximum two candidate slots per event**,
the same 10 ms *assumed* deadline, the same ~8 GB/s **assumed** upload rate,
and the same two physical NVMe read workers. Experts already listed in
host RAM or GPU residency were skipped for disk reads. One 4 MiB aligned
window is read from each candidate's **actual** mapped up-projection
tensor. This is not the complete ~14 MiB expert and not a GPU transfer.

| Frozen policy | Predictions useful | Native expert-selection recall | Physical SSD bytes per run |
|---|---:|---:|---:|
| Previous-token reuse | 36.04% | 9.01% | 3.859 GB |
| Cross-layer transitions | 41.34% | 10.34% | 3.817 GB |
| **Causal fusion** | **47.29%** | **11.82%** | **3.859 GB** |

A first full run evaluated policies previous-token / cross-layer / fusion;
the independent second physical pass reversed that order. Every policy
returned the same usefulness/recall across passes (since the frozen
held-out labels and policy were the same). The logged Linux block-device
sector-counter read deltas equaled each experiment's expected 4 MiB
`O_DIRECT` byte volume. **Total data read across both passes: 23.069 GB.**

| Policy | p99 two-reader completion, first pass | Reverse pass |
|---|---:|---:|
| Previous-token | 3.081 ms | 3.350 ms |
| Cross-layer | 3.283 ms | 3.144 ms |
| **Causal fusion** | **3.026 ms** | **3.246 ms** |

The 96-event initial pilot had 70.31% fusion precision, but that falls
to 47.29% over the full 462-event held-out sequence; report the full-run
figure when choosing policy. Raw receipts capture individual read p50/p99,
two-reader p50/p99, disk-sector counters, candidate budgets and trace hash.

**Not a measured GLM speedup**: 10 ms layer slack and 8 GB/s upload
were budget assumptions, not observed concurrent compute/upload.
The experiment issues speculative reads **after freezing the predictions,
but outside actual inference**. It does not retain tensors in RAM/VRAM,
avoid a native SSD miss, evict a cache entry, overlap router compute,
alter attention/RoPE or produce tokens. Sensor safety margins and
concurrent multi-session admission remain separate release gates.

## Reproduction and protections

`benchmark_trace_predicted_reads.py` maps the real safetensor headers
to physical offset, then runs two-reader `O_DIRECT` requests under
a 384 MiB systemd hard cap, **no swap**, low IO/CPU priority, and a
host-memory PSI abort. It refuses missing provenance, unsupported
cache bitmaps, invalid source shard, out-of-bounds offsets and missing
candidate tensors.

Run 462-event held-out test with the two policy orders (separately;
only while production host PSI is low), passing `--execute-io`.
Omitting that flag is a dry-run with **zero SSD reads**. Tests in
`c/tests/test_glm53_trace_predicted_reads.py` prove changing the current
route labels cannot change a prediction; stale cache generations and
missed deadlines fail closed. The existing seven tests in
`test_glm53_shadow_tier_plan.py` also pass.

## The safe next step

1. Expand representative **real task** traces across independent runs;
   the existing synthetic-token holdout is too narrow to establish
   future usefulness. Keep private activations out of the repo.
2. Log actual time-to-route and Vulkan upload p95/p99 inside an isolated
   model call. Use measured, not assumed, transfer budgets.
3. Implement a finite **QD2 reader queue** with speculative page ownership
   and cancellation on session/generation change; reserve memory first.
   Compare native misses avoided per byte against no-prefetch and simpler
   predictors, with exact native token/logit parity.
4. Because full GLM on Joyner has repeatedly stalled due to host RAM
   pressure, keep live speculative imports disabled until memory pressure
   is resolved or benchmarking moves to a higher-RAM node.

**Production: HOLD.** The experiment did not modify inference services,
AssistX routing, model weights or any runtime cache. Both Joyner model
endpoints remained HTTP 200 throughout.
