# GLM-5.3 exclusive R9700 hardware benchmark — October 17, 2026

Status: experimental full-model runtime, production fleet admission HOLD.

## Reversible isolation procedure

Both known non-admitted R9700 inference services previously held the same physical GPU render device while GLM benchmarking occurred:
- fleet-r9700-vulkan-canary.service (MiniCPM; port 18320)
- fleet-r9700-toolcall.service (3B tool calling; port 18321)

The exclusive controller checks host and user identity, exact GPU device mapping, canary MainPIDs, endpoint health, GPU handles and unexpected heavy processes. It acquires a nonblocking lock. Unknown GPU users cause refusal, not arbitrary termination.

The controller arms an independent four-minute systemd recovery timer before stopping those two exact user services. It verifies no other process holds the GPU, then runs the selected GLM tests under explicit memory, swap, GPU, PSI, and time limits.

On completion or error, owned GLM scope units are stopped before restoring the canaries. Both original endpoints must return HTTP 200 before the recovery timer is cancelled. A separate recovery script handles unexpected controller termination.

Only these non-admitted canaries may be temporarily interrupted. Networking, storage, container, OS, and unrelated production services are left unchanged. The controller must be separately authorized if these canaries are promoted to critical production service.

## Full-model physically isolated A/B

| Metric | Control promotion | Fair-share promotion |
|---|---:|---:|
| Decode throughput | 0.376 tok/s | **0.403 tok/s** |
| Routed experts served by GPU | 418 / 3,696 | **565 / 3,696** |
| SSD expert loads | 3,114 | **2,961** |
| GPU expert uploads | 326 | **270** |
| Model RSS | 3.38 GiB | 3.38 GiB |
| Lowest available host RAM | 8.73 GiB | 8.59 GiB |
| Maximum memory PSI full.avg10 | 0.00 | 0.00 |
| Generated output | identical | identical |

The fair-share policy improved measured decoding by 7.18% in this single paired full-checkpoint run, with 35.17% more expert selections served by GPU, 153 fewer SSD expert loads and 56 fewer GPU uploads. This does not establish a statistically reliable multi-prompt or production performance gain.

The first exclusive attempt was cancelled at four seconds by an incorrect endpoint-health check; both canaries were restored. The watchdog was corrected and covered by five tests before the successful pair.

Following the second pair, both serving endpoints again returned HTTP 200, both user services were active, no GLM benchmark survived, and the recovery timer was cancelled.

## Reproduction

On Joyner, use the pinned shadow directory. First invoke exclusive_joyner_benchmark.py with --dry-run; then invoke it normally for the two approved cases. Each result uses a new exclusive-create filename.

Tests: test_glm53_competition_guard.py (five checks); test_glm53_benchmark_isolation.py (four checks). Aggregate receipts live in hardware-20261017/exclusive-r9700.

The GLM production admission decision remains HOLD.
