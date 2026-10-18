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

## Stronger CPU, SSD and GPU exclusion (2026-10-17 follow-up)

The GPU-only check was insufficient for a full-model performance claim. The
new glm53_competitors.py uses monotonic, paired /proc readings to measure
per-process CPU ticks, block read_bytes+write_bytes and resident RAM;
process starttime detects PID reuse. The two known canary service PIDs are
excluded *only* during the pre-stop inventory; once they have been stopped,
the exclusive runner expects them to remain stopped.

A benchmark is refused if a competitor uses >=0.80 CPU cores or >=32 MiB/s
of disk read+write bandwidth, or holds >=2 GiB RSS. The new preflight also
samples aggregate /sys/block/nvme0n1/stat; >=32 MiB/s of background NVMe
traffic is a NO_GO before launch. During GLM execution, global block traffic
includes the benchmark itself, so the periodic guard instead checks other
processes and the exact R9700 GPU device holder list every ~4 seconds.

The trial's own process group is excluded only by its exact, transient
systemd scope name. An unexpected GPU holder, heavy user/system process,
or restart of either known canary triggers watchdog cleanup, NOT a
contaminated speed result. The exclusive controller removes leftover scopes
from its own exact GLM benchmark namespace before starting, after taking
its exclusive lock. It never generically kills a by-name match or a
network/storage/OS service.

I/O attribution has known limits: on Joyner, /proc/PID/io was inaccessible
for 416 of 495 scanned PIDs (mostly system/kthreads/other users), although
the process identities and CPU counters were visible. The aggregate NVMe
sample was idle, and a controlled synthetic CPU competitor was detected.
This does **not** prove zero privileged background disk I/O during each
forward: it is an explicit measurement limitation rather than a success
claim. In a more tightly controlled lab, stronger per-cgroup I/O or block
telemetry is necessary.

Validation: 12 process-scanner unit tests, 7 watchdog tests, 4 restoration
tests pass. The latest dry run inspected 495 processes, found no unexpected
high-activity process or NVMe work, left the two canaries running, and
verified both endpoints returned HTTP 200. No new full-model throughput
measurement was run for this guard-only change.

## Automatic termination of registered disposable competitors

The exclusive controller now actively stops competing user-managed jobs in
explicit disposable systemd service/scope namespaces: glm53-disposable-,
colibri-disposable-, opencode-disposable-, or hermes-disposable-, each followed
by exactly 12 lowercase hexadecimal characters and a .service/.scope suffix.
This identity, process UID and cgroup membership must all match. Mere
process-name resemblance, high CPU, or an untrusted arbitrary PID is never
sufficient authorization to terminate a service.

Before stopping canaries, the controller measures CPU usage, SSD I/O, memory
usage and unexpected R9700 owners, stops recognized disposable competitors,
and verifies their unit ownership has gone away. It repeats the inspection
after the GPU is cleared. During the benchmark a newly detected competitor
of this type is terminated, but the **affected trial is still invalidated**
and its termination is included in the failed trial receipt. Unrecognized
CPU/GPU/SSD contenders cause an immediate NO_GO, and system infrastructure,
user sessions and network/storage services are not terminated.

These stop operations are audited in a private append-only JSONL receipt
with the approved unit, PID, observed resource contention and time. The
controller dry-run performs no termination. A new exclusive invocation
generates a unique 12-character round identity so existing receipts are
never overwritten, and two runs of the same binary remain paired by
that identity.

Physical proof on Joyner: first, an explicitly tagged disposable service
was stopped by its exact unit and its original process exited. A second
CPU-saturating disposable service was then measured at about 1.0 CPU core
and **automatically terminated by the controller's contender sampler**.
Both active Tailnet inference endpoints stayed HTTP 200; no production
application or infrastructure unit was stopped in those tests.

The relevant tests include nine synthetic process-identity/termination
controls, including cross-user, production-unit, lookalike, dry-run and mixed
safe/unsafe PID denial. All exact units must be registered as disposable
by their launch configuration; ordinary unregistered OpenCode/Hermes
workloads cannot be silently destroyed.

The full-model benchmark runner now accepts unique paired names of the form
exclusive4control-<12 lowercase hex> and exclusive4fair-<12 lowercase hex>
only with --isolated and only at the pinned 4 GiB / 32-promotion configuration.
Two additional tests verify unique round labels cannot bypass GPU isolation.
The old results are preserved and remain independently auditable.

## Follow-up paired real-checkpoint benchmark — 2026-10-17 20:47–20:50 EDT

The exclusive controller completed an additional physical same-binary A/B
on Joyner with both R9700 canary services stopped, after a clear CPU, GPU and
Gen5 SSD preflight. Unique round ID: a5c017204700.

| Measurement | Control | Fair-share |
| --- | ---: | ---: |
| Decode throughput | 0.373 tokens/s | 0.400 tokens/s |
| GPU experts served / 3696 | 418 | 565 |
| SSD expert misses | 3114 | 2961 |
| GPU uploads | 326 | 270 |
| Engine RSS | 3.38 GiB | 3.38 GiB |
| Lowest host MemAvailable | 8,863,140 KiB | 8,847,288 KiB |
| Maximum memory PSI avg10 full | 0.18 | 0.00 |
| Detected competitors | 0 | 0 |
| Greedy output | 16 11 21 13 17 11 21 13 | exactly identical |

The fair-share configuration was 7.24% faster in this pair using pinned
binary SHA-256:
7107c88241266da8bfcbb7ba82683dc93b4bca53ee7fd54584d2813f5ca247d1.
This is an additional single matched pair, not production-ready statistical
evidence. It corroborates the earlier isolated 0.376 -> 0.403 tok/s result
while showing full-checkpoint interactive speed remains unacceptably low.

Physical controller outcome: EXCLUSIVE_RESTORE True; both original canary
endpoints 18320 and 18321 returned HTTP 200 after recovery, with new healthy
PIDs, no orphan GLM systemd scope, no restoration timer, and memory pressure
avg10 back to 0. No other disposable CPU/SSD/GPU workload required
termination in this run. The only known competing processes were the two
canaries explicitly stopped and restored. Raw JSON receipts are:
- benchmarks/joyner-glm53-20261016/exclusive4control-a5c017204700.json
- benchmarks/joyner-glm53-20261016/exclusive4fair-a5c017204700.json

GLM production admission remains HOLD.
