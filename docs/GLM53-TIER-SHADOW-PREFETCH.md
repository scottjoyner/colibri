# GLM-5.3 tier-aware shadow prefetch — draft rollout

This fork-only experiment **does not issue SSD reads, GPU uploads, or native
router changes**. The existing model and production endpoints are unchanged.

## Independent-run evidence

The GLM checkpoint is pinned to source revision
`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.
Runs 01–03 fit historical expert transitions, run 04 chooses fusion weights,
and runs 05 and 06 remain untouched holdouts. Activation traces and trained
head weights stay in a private fleet directory, not in Git.

The resulting frozen fusion weights are:
previous-token 1.0, cross-layer 2.0, layer frequency 0.25.
Held-out native-expert selection recall at 16 candidates:

| Predictor | Run 05 | Run 06 |
|---|---:|---:|
| Previous-token baseline | 37.5% | 41.1% |
| Cross-layer baseline | 26.9% | 46.3% |
| **Causal fusion** | **43.2%** | **55.3%** |
| Attention head v2 hybrid | 36.0% | 36.8% |

These short synthetic-ID prompts do not prove generalization to real tasks.
The first learned attention heads remain below deterministic controls.

## Bounded, tier-aware shadow simulation

The planner refuses mismatched source pins, stale cache generations, missing
session IDs, v1/no-inventory traces, unknown bandwidths, insufficient
deadlines, and GPU/RAM byte quotas. It skips experts already in VRAM and
distinguishes host-resident experts from unknown disk/page-cache candidates.
It never consults the current native routing label when proposing experts.

Under a hypothetical 10 ms window and two candidate slots per step, the
run-06 counterfactual showed **47.3% useful proposals** for causal fusion,
covering **11.8% of the selected native experts**, and about **13.0 GB
of nominal candidate weights** over the 462 routed events. Previous-token
reuse scored 36.0% useful proposals and 9.0% coverage at similar nominal
byte totals. None of these bytes was physically transferred.

The simulator uses a previously measured 9.496 GB/s short NVMe expert-read
rate; **the assumed 8 GB/s upload speed and 10 ms window are NOT measured**.
These results cannot be used to claim lower latency or SSD read volume.

## Why not activate DFlash2 yet?

[DFlash](https://proceedings.mlr.press/v306/chen26bv.html) and
[DFlash2](https://github.com/z-lab/dflash) draft and verify *tokens*.
The currently listed DFlash2 checkpoints are not GLM-5.3. We borrow the
lookahead + native verification boundary for *expert weights*, not token
acceptance. Every authoritative expert choice and logit remains native.

[SpecMD](https://proceedings.mlr.press/v306/hoang26d.html) suggests
Least-Stale cache policy comparisons; [MoE-SpeQ]
(https://arxiv.org/abs/2511.14102) motivates bounded expert lookahead;
[SpecPrefetch](https://arxiv.org/abs/2607.24787) explicitly retains the
original router's selection while speculating only about transfers.

Before moving from shadow to a live canary: collect representative held-out
prompt workloads and multiple sessions, independently measure PCIe/Vulkan
upload p95 and physical SSD/page-cache misses, test capacity and eviction
simulation, calibrate confidence, prove cancellation/generation fencing and
exact token/logit parity. Keep production admission **HOLD**.

