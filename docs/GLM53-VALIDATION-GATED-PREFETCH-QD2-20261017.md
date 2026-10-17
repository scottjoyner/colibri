# Colibri GLM-5.3 — bounded validation-calibrated QD2 shadow prefetch

**Date:** October 17, 2026. **Node:** Joyner / Samsung 9100 PRO Gen5 / AMD R9700.
**Scope:** independently held-out native GLM expert route replay with actual O_DIRECT reads;
**not** model inference, GPU staging, speculative routing, or production prefetch.

## Next optimization completed

PR #10 showed that two-reader causal-fusion predictions matched **47.294%**
of candidates on a 462-event held-out run, but emitted 924 predicted experts,
920 of them requiring unknown-tier SSD reads. This patch introduces a
**validation-calibrated second-candidate gate** to reduce wasted reads.

Native routing training uses independent runs 01–04. Run05 is held out
from training and used **only** to estimate the conditional success
probability of a second ranked candidate per layer. The second candidate
is kept if its layer's Bayesian-smoothed validation success probability
is at least **0.40** (prior strength eight events). This threshold was
fixed before reading run06-v2, the final independent 462-event evaluation.
The first candidate is never suppressed by this gate.
No native labels from run06 are inspected until after the plan is frozen.

All 62 GLM converted safetensor shards, source SHA pin
`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`, and 4KiB-aligned
tensor offsets were checked before physical reads. Same 4 MiB source
up-projection windows and 10 ms **assumed** layer deadline for each policy.

## Two real SSD A/B comparisons, opposite test order

| Held-out metric | Frozen fusion, two candidates | Frozen fusion + validation gate |
|---|---:|---:|
| Predicted candidates | 924 | **715** |
| Suppressed second nominations | 0 | **209** |
| Candidate precision | 47.294% | **49.930%** |
| Native expert selection recall | **11.824%** | 9.659% |
| O_DIRECT bytes each full pass | 3.8588 GB | **2.9947 GB** |
| SSD data avoided per pass | — | **0.8641 GB / 22.39%** |
| p99 two-reader step, baseline→gated | 3.139 ms | 3.176 ms |
| p99 two-reader step, gated→baseline | 3.152 ms | 3.190 ms |

The Linux Gen5 block-device sector-counter deltas matched each
mode's O_DIRECT transfer byte count in both passes. The second physical
run reversed the policy order. Both ran in a systemd unit with
`MemoryMax=384M`, no swap, low CPU/IO priority, and safety checks.
The same fixed held-out route is replayed twice; this is not two
independent inference task distributions.

### Important loss-of-coverage tradeoff

The gated policy issues **22.62% fewer nominations** and reduces SSD read
volume by **22.39%**, but native expert-selection recall drops **18.31%
relative** (from 11.824% to 9.659%). Higher precision is not automatically
better: this would be a **bandwidth/latency constrained profile**, not a
replacement for maximal recall under unlimited I/O.

## Bounded shadow reader

`bounded_shadow_reader.py` adds a separate, opt-in *test harness*
containing a session/40-char source pin/generation owner, exactly two
O_DIRECT readers, and one 8 MiB in-flight memory reservation per batch.
It rejects a concurrent second batch or mismatched owner; revocation
prevents new admissions. An active native read cannot safely be
interrupted, so errors drain active futures before returning. An
in-flight generation revocation makes the result ineligible for reuse.

A final 64-event physical read check ran the revised reader
successfully, with both production model endpoints returning HTTP 200.
Six bounded-reader/validation tests, four existing route-leakage/alignment
tests, and seven shadow-planner tests passed on x1-370: **17 tests total**.

## Explicitly unresolved

- This physical test reads only **4 MiB** of the expert's up-projection
  and never uploads the complete ~14 MiB expert to the GPU.
- Candidate usefulness reflects synthetic-ID GLM traces; need
  representative independent task traces before trusting its rate.
- The deadline assumes 8 GB/s upload, rather than measuring overlap
  with live attention or router evaluation.
- The final full-checkpoint GLM canary on Joyner is still blocked by
  system memory-pressure stalls; do not enable imported-host or
  speculative caching in the active inference service.
- No usable task-level token/sec gain, native logit parity under
  prefetch, multi-session isolation or node-wide storage QoS is proven.
- The reader is a **synchronous shadow executor with two parallel reads
  per event**, not a live asynchronous inference prefetch implementation.

Raw physical results: `hardware-20261017/calibrated/` (full A/B and
reverse B/A) with source trace hashes and block-sector read deltas.
The final 64-event check has separate receipts. None contains the
private activation trace itself.

**Decision:** retain as a draft, disabled experimental shadow capability.
The next GO gate requires captured real layer slack, bounded live GPU
upload and exact token/logit parity under memory pressure safeguards.
