# GLM-5.3 fleet runtime — read-only preflight, Joyner

**October 17, 2026.** Experimental candidate, stacked after Colibri PR #14.
**Current outcome: HOLD / NO-GO for production GLM model admission.**

This is an intentionally **read-only** check of the already-converted
62-shard GLM-5.3 checkpoint, current host memory pressure, and physically
identifiable Radeon AI PRO R9700 VRAM. It does not start inference, unload or
reserve another model, change fleet model routing, drop page cache, modify
weights, or open an externally reachable listener.

## Verified physical checkpoint and node

`python3 benchmarks/joyner-glm53-20261016/glm53_fleet_readiness.py
--model /srv/joyner-gen5/models/glm53_colibri_i4` (on Joyner).

| Category | Actual checkpoint bytes | Tensor count |
|---|---:|---:|
| Routed expert weights | 175,305,129,984 | 74,304 |
| Other dense weights | 17,015,135,608 | 1,270 |
| Output | 1,268,776,960 | 1 |
| Vision | 1,127,927,808 | 347 |
| **Logical total** | **194,716,970,360** | **75,922** |

The scan inspected safetensor *headers* only, without touching tensor
payloads. It validated every shard's exact canonical name, ordinary
regular-file identity, tensor ranges and non-overlap, descriptor stability
and six complete tensors for all 12,384 experts. Aggregate header
SHA-256: `a4d728e48359108beca382eb34a0656804f4fb19df007ee5589c1cd1000e1231`.
This is a **header-integrity witness**, **not** a cryptographic validation
of the 194.7GB tensor content or of upstream revision
`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`. Both remain independent
release gates.

At observation time:
- Joyner physically reported ~14.75GiB total and ~5.9GiB available RAM.
- **1,098,723,328 swap bytes in use**; no live memory PSI pressure.
- R9700 identified as AMD PCI ID `1002:7551`, `card1`; 34,208,743,424
  VRAM bytes total and 59,912,192 used in the sysfs snapshot.
- Gen5 SSD had approximately 666GB free.
- Existing authenticated Tailnet endpoints on ports 18320 and 18321 both
  returned HTTP 200 before/after. They serve other models, **not full GLM**.

The preflight compares declared minimum host+GPU reserves and refuses
missing GPU identity, swap usage, or memory pressure. Even when those
numbers are otherwise acceptable, it **always** returns `decision=HOLD`.
It cannot itself authorize execution.

The failed prior Joyner full-model attempts have been read back from
systemd: one timed out (exit 124), the other was OOM-killed. The
current preflight does not override that negative operational evidence.

## Actual next experiment and release gates

1. **Resource isolation:** first establish a bounded *dense-loading* path
   on a machine with enough memory headroom without affecting Joyner's two
   active services; measure full startup peak RSS, OS PSI, GPU memory,
   temporary SSD writes, and cancellation cleanup. Bounded checkpoint
   reads and a healthy tiny fixture are not adequate full-model proof.
2. **End-to-end payload parity:** integrate native grouped/preadv expert
   reads, imported-host Vulkan transfer, and all six tensor upload offsets
   into one controlled path. Verify direct GPU readback for 1/2/4-run
   layouts, actual GLM logits and generated tokens (not only tiny fixtures).
3. **Latency admission:** measure real router-to-expert wait windows,
   p50/p95/p99 layer time, time-to-first-token, sustained tokens/s and
   memory pressure for representative natural prompts.
4. **Cache correctness:** session-bound leases, generation fencing,
   cancellation, eviction and source freshness before enabling any
   prediction-read policy. The gated PR #12 heldout predictor has low
   selected-expert recall: **9.659%**.
5. **Fleet integration:** authenticated endpoint qualification, per-user
   receipts, available-vs-loaded distinction, stale-effect rejection,
   offline behavior, bounded canary and explicit rollback. No automatic
   AssistX/Auto-Router enrollment of experimental GLM.

## Controls and repeatability

The stdlib-only script reads `/proc/meminfo`, `/proc/pressure/memory`,
`/sys/class/drm`, and every checkpoint shard's metadata with read-only
file descriptors. It never reaches out to network services. For tests,
`--sys-root` makes proc/sys inputs injectable without manipulating a
production host. On unexpected input the script returns exit 2 with
`decision=NO_GO`; on a successful audit it returns exit 0 but the **JSON
decision remains HOLD**, not PASS. Do not mistake successful completion
for approval to start inference.

The exported receipt contains no model tensor values, private routing
traces, API keys or per-expert source offsets. Eight focused offline
tests cover PCI device identity, impossible VRAM measurements, meminfo,
memory PSI, checkpoint absence, incomplete shards, invalid budget values
and swap accounting. Production remains untouched.

**Do not deploy the GLM service, auto-admit to the fleet, or restart
production llama.cpp endpoints from this preflight.**
