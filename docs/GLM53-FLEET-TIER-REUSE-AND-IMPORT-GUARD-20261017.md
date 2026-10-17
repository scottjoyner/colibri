# GLM-5.3 fleet runtime: causal expert-tier admission and imported-host safety

**October 17, 2026. Experimental / production HOLD.**
Stacked on PR #16. No model deployment, service restart, private trace publication,
or change to the two running Joyner inference servers.

## First: restore the successful full-model baseline in the release record

The physical [full-GLM import A/B report](GLM53-REAL-INFERENCE-HOST-IMPORT-AB-20261017.md)
already demonstrates a **completed eight-token full-checkpoint run** with
bounded dense staging, 45 GPU dense-chain layers, 6 GiB process RAM
limit and staged Vulkan host **copies**, `COLI_VK_UP_IMPORT=0`.
The run decoded eight native greedy IDs in 12.7 s (**0.630 tok/s**);
3,299 expert loads were counted, 231/3,696 routed selections ran on the
GPU, and 1,552 promotion attempts were skipped by the per-token rate limit.
It remains a **non-production** synthetic-prompt reference, not a natural
dialogue quality or sustained throughput claim.

The paired import-on run (`COLI_VK_UP_IMPORT=1`) never completed; machine
PSI full avg60 reached ~20.59%, and MemAvailable collapsed to ~199 MiB.
It was operator-stopped. These figures do not identify import churn as
the unique root cause, but rule out promoting that unsafe configuration.

## Runtime safeguard: low-host-RAM Vulkan host-import guard

New pure decision helper `c/glm53_import_safety.h` and native GLM startup
integration refuse dangerous imports for machines with **<=16 GiB of
total RAM**, or when total host RAM cannot be measured:

- `COLI_VULKAN=1 COLI_VK_STAGED=1` with no
  `COLI_VK_UP_IMPORT`: force `COLI_VK_UP_IMPORT=0`
  **before model loading**, i.e. use bounded staging copies.
- Explicit `COLI_VK_UP_IMPORT=1` without the *separately named*
  `GLM53_HOST_IMPORT_EXPERIMENT=1`: **fail closed**, exit 2.
- Explicitly disabling import remains supported. An experimental override
  requires **both** settings; the engine prints a strong warning.
  That override is NOT fleet admission. On larger-RAM hosts, existing
  defaults remain unchanged; other engines are unaffected.
- Linux / Vulkan only. Embedded GLM also checks before GPU initialization.

**Real Joyner verification:** an attempted import-on launch with a missing
model path was denied by the guard before reading the model. A separate
tiny-GLM Vulkan run with no explicit import flag forced staging copies
and returned the exact expected greedy tokens `34 121 34`. Production
ports 18320/18321 were still HTTP 200 and PSI was zero.

## Experimental causal repeat-first promotion, default OFF

`COLI_VK_TIER_REPEAT_ADMIT=1` affects **GLM-5.3 only**. It records
how many **distinct forwards** selected each (layer, expert), independent
of saved routing history; promotion can be considered only after two
different forward epochs. Repeated prefill rows in the *same forward*
do not count twice. The epoch is monotonic, resets safely after wrap,
and never authorizes a cache read or dispatch: it only suppresses some
copies to the asynchronous Vulkan promotion queue. It also applies to
`vkt_wants()` and increments read-only skipped-admission telemetry.

This does NOT change the existing `COLI_VK_TIER_RATE`, eviction score,
generation state, input logits, attention, quantization or default tier
placement. Actual uploads still require the engine's existing checks:
device available, room/eviction score, rate cap, queue admission, and
completion of asynchronous upload/fence. No cross-session sharing added.

### Real tiny-fixture A/B on Joyner R9700

Both modes exercised native vectorized expert reads; token sequences
and teacher-forcing scores match byte-for-byte.

| GPU upload mode | Existing | Repeat-first |
|---|---:|---:|
| Ordinary Vulkan tier GPU uploads | 3 | 2 |
| Ordinary Vulkan tier GPU-served selections | 6/22 | **2/22** |
| Repeat-first one-off admission skips | 0 | 3 |
| Staged import-disabled tiny run: uploads | 0 | 0 |
| Staged tiny output parity | PASS | PASS |

**Negative evidence:** repeat-first saves uploads but reduces useful
GPU service in this tiny test. Do **not** claim a speedup or turn it
on fleet-wide on that basis.

### Native routing-trace causal simulator

The 462-event native GLM holdout corresponds to 42 sparse layers,
9 forward frames (3 prefill rows followed by 8 decode forwards),
3,696 selected labels and 1,615 unique (layer, expert) labels.
An **independent 798-event native trace** is used for validation.
All policy selection uses validation only, never holdout.

The simulator is optimistic: promotes at the END of a forward with
zero modeled latency; a later forward can use a resident expert.
It counts exactly one visit per expert per distinct forward and
never predicts future expert identities. It omits real GPU driver
fences, async backlog, 256-MiB pool fragmentation, cache warmstart
timing, CPU reuse and end-to-end kernel cost.

| Configuration in simulator | Holdout GPU hits / 3696 | Simulated GPU uploads |
|---|---:|---:|
| 80 slots, 16/token, first-eligible | 120 (3.2%) | 432 |
| 80 slots, 16/token, repeat-first | 348 (9.4%) | 384 |
| 320 slots, 32/token, first-eligible | 534 (14.4%) | 864 |
| 320 slots, 32/token, repeat-first | 791 (21.4%) | 768 |
| 320 slots, 32/token, repeat-priority but no gate | 865 (23.4%) | 864 |

These are **not** native GPU hit measurements. The simulator's
320-slot cap represents only **4.22 GiB of raw expert tensors**, not
an independently verified Vulkan tier budget, which may be larger
because of pool fragmentation and streaming slots.

Validation policy objective was fixed as simulated hits minus
0.30 times GPU uploads, for capacities <=320 and upload rate <=32.
Validation selected (320 slots, rate 32, repeat-first); the
heldout values above are reported only afterward.

### Next fleet runtime actions (not yet approved for production)

1. Validate a 320-slot/32-rate variant in a **bounded** physical
   GLM native run on a host with sufficient RAM; do not equate 320
   logical slots to a particular `COLI_VK_TIER_GB` setting.
2. Carry forward `COLI_VK_UP_IMPORT=0` and the successful bounded
   dense 45-layer baseline. Monitor cgroup memory/PSI, actual VRAM,
   upload bytes/rate skips and both existing service health endpoints.
3. Compare native exact tokens and logits, p50/p99 expert wait,
   prefill latency, TTFT and sustained tokens/sec against 0.630 tok/s.
4. Reject candidate on poorer GPU service, higher SSD bytes,
   memory reclaim stalls, live endpoint impairment, or parity errors.
5. Add per-user route receipts, session fencing, cancellation and
   source revision checks before admission to AssistX or Auto-Router.

**Production rollout, model service and model registry remain unchanged.
No performance uplift is claimed.** This draft retains the feature
behind an opt-in until the true full-model results exist.

## Reproducibility

- `c/tests/test_vk_tier_repeat.c`: distinct forward counts, repeated
  rows, no-future admission, no stale label carryover.
- `c/tests/test_glm53_import_safety.c`: explicit import denial, safe
  staging copy default, override, unknown RAM, large host and NaN.
- `c/tests/test_glm53_native_tier_tuner.py`: seven offline policy
  invariants, incomplete trace rejection and reproducibility.
- `benchmarks/joyner-glm53-20261016/tune_native_expert_tier.py`:
  deterministic, validation-fitted causal simulator. Reads native
  private trace files *on the user node*, emits only aggregate metrics.
- `benchmark_repeat_tier_fixture.py`: R9700 tiny native parity and
  direct tier telemetry in two GPU staging modes.
- Aggregate outputs in
  `benchmarks/joyner-glm53-20261016/hardware-20261017/tier-policy/`.
  Raw routing predictions, private offsets and model weights are NOT
  included in repository artifacts.

Rollback: leave `COLI_VK_TIER_REPEAT_ADMIT` unset. For the import guard,
set `COLI_VK_UP_IMPORT=0` (the known successful staged baseline).
No automatic fleet deployment has occurred.
