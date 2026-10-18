# GLM-5.3 | selective expert prefetch and the 24/7 operating envelope

Status: HOLD — shadow-only design, not a serving optimization.
October 18, 2026. The R9700's last verified single-row streaming A/B is
0.750 tok/s control vs 0.273 tok/s streamed (-63.6%); 2,961 vs 3,424
expert reads and 565 vs 2,688 GPU-served experts. The increase in GPU
work did not reduce decode latency. Do NOT enable the opt-in
COLI_VK_TIER_SINGLE_ROW_STREAM flag on a production service.

## What was implemented

The selective_prefetch_gate.py module is a pure, read-only admission
decision layer atop existing shadow_tier_plan.py and independent-run
calibration machinery. It does not issue SSD reads, allocate VRAM,
evict resident experts, dispatch Vulkan, start servers, or admit models.

A shadow nomination requires all of:

1. Exact model pin, session and lease generation shared by caller,
   calibration model and current hardware witness.
2. Fresh monotonic resource snapshot (<=1,500ms old); R9700 and SSD
   healthy and memory PSI full avg10 == 0.
3. Independent validation: >=2 source runs, >=30 validation events,
   split verification, >=70% observed precision, and each candidate's
   independently estimated usefulness probability >=70%.
   These are provisional research limits, not verified calibration.
4. Free VRAM after a 1GiB emergency margin, reserved app capacity
   and inflight speculative buffers. No reclaiming resident experts.
5. A nonfull queue, bounded SSD/GPU budgets (<=2 x ~13.5MiB per event),
   and a bounded estimated deadline.
6. An exact, valid 288-expert v2 GPU/host residency inventory.

The planner cannot consume current routing labels. A synthetic test
verifies changing the native evaluation labels has no effect on plans.
Missing, stale, invalid or low-confidence evidence means zero proposals.

CRITICAL LIMITATION: The pure planner does not guarantee no eviction
inside the future Vulkan execution engine. A native implementation
must require a generation-scoped reservation, explicitly separate
speculative GPU capacity, cancel stale work, and produce transfer
receipts. Until then, all proposed nominations are nonexecuting.

## Validation and limitations

- 17 targeted unit tests cover confidence, resources, lease fencing,
  device health, memory pressure, free VRAM, queue pressure, masking,
  byte limits, and no native-output-label peeking.
- The six independent native GLM expert routing traces remain on Joyner.
  The node is currently unreachable by SSH and was reported offline
  by Tailscale. No physical GPU or real-held-out trace replay ran on
  October 18. There is no evidence yet of faster inference.
- This read-only module is a research safety primitive, not a service.

## 24/7 acceptance constraints

Availability: Joyner offline is a P0 operational warning. Inference
must fail over to a separately admitted worker using independent
model-loaded and synthetic-generation health checks. Distinguish
DISCOVERED, LOADED, UNREACHABLE and NOT ADMITTED states.

Resource protection: bounded nonroot systemd services, memory/VRAM
limits, queue depth, SSD writes and read pressure, graceful shutdown,
per-user quotas, overload shedding and no reliance on uncontrolled
swap or checkpoint modifications.

Security: authenticated owner-bound admission, two-user separation,
revocation, exact execution receipts, stale lease fencing, fail closed
if control plane loses quorum.

Correctness and performance: reference logits, long and natural prompts,
three controlled isolated A/B pairs per binary, p50/p95 latency, resource
and energy accounting, repeatable rollback, fault injection and
72-hour soak under real concurrency.

Admission: glm53_production_gate.py remains read-only and always HOLD.
An HTTP 200 code by itself does not establish a loaded, working model.

## Next acceptance experiment

1. Verify independent Joyner health via normal fleet operations.
   Do not blindly reboot or terminate unknown processes.
2. Run shadow-only scoring against held-out run06 v2 with training
   on runs01–04 and threshold fitting on independent run05.
3. Measure cold-hit precision, wasted bytes, predicted deadlines,
   GPU-resident eviction avoidance, and effect on I/O headroom.
4. Only after positive data, explore a bounded Vulkan implementation
   on a separate default-OFF branch, with exact output parity,
   reversible tests and no production routing.
