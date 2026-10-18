# Colibri GLM-5.3: 24/7 production acceptance (October 17, 2026)

Status: **HOLD**. This is a production-readiness contract, not a deploy order.

## Two separate gates

Performance and safety are independent requirements. The Joyner 321B
checkpoint generates eight verified output tokens at ~0.400 tokens/s with
fair-share expert promotion. The bottleneck is cold MoE expert movement:
in a paired 8-token run, 2,961 SSD expert loads moved 39.0 GiB while
only 565/3,696 routed experts ran on the R9700. This is not interactive
production throughput. Improvements must preserve per-token routing,
output parity and an operator-controlled fallback.

### Gate A: throughput and correctness

- At least three independently measured, isolated A/B pairs per binary,
  under the published reversible benchmark controller, plus reversed
  execution order and representative prompt diversity.
- No concurrent CPU/GPU/SSD contenders, no memory-floor or pressure
  watchdog violations, no orphaned benchmark scopes.
- A first interactive engineering target of >=3 tok/s under the pinned
  workload; performance against longer, real-world prompts with prefill,
  decode p50/p95, energy/token and sustained throughput must follow.
  Targets are NOT predictions or safety proofs.
- Reference logits and sampled/greedy completions against known-correct
  checkpoints, not just identical output across two experiment flags.
- New single-token expert GPU streaming must be opt-in and abort safely on
  allocation, upload, dispatch or synchronization failure.

### Gate B: continuous operation and security

- At least 72h live serving with >=1,000 requests, >=100 natural-language
  requests, <=0.1% failed requests and p95 token latency <=750ms for the
  interactive workload; repeat after major driver, model or backend changes.
- Verified host RAM and VRAM margins under concurrent load with a
  fail-closed deadline, admission backpressure, request cancellation and
  bounded queue. Never let swap or allocator exhaustion destabilize Joyner.
- Demonstrate process crash recovery and cold restart <=60s where feasible,
  and test GPU disappearance/driver error, SSD read failure, corrupt shards,
  out-of-space, memory pressure, Tailnet loss and graceful shutdown.
- Authenticated only through approved AssistX/Auto-Router/Tailscale pathways;
  isolation across simultaneous users, forged caller denial, token
  revocation, rate limiting, audit trails and tenant separation verified.
- Health probes include model loaded (not just discovered), prompt
  generation ability, backend device identity and exact binary/model
  fingerprints. Health green cannot be inferred solely from an HTTP 200.
- Prove versioned release, explicit rollback, stale lease fencing,
  credential/key custody and fail-closed operation if control plane fails.
  Rollback must not overwrite the authoritative 194GB+ model shards.

## Read-only release scorecard

Run glm53_production_gate.py with physical benchmark JSON receipts.
The optional soak JSON schema is validated for obvious inconsistencies; it
is untrusted self-reported evidence, not an attested proof. Independent
signed source/checkpoint and operational witness verification must still be
performed, and the scorecard NEVER issues deployment authorization.

Command:
python3 benchmarks/joyner-glm53-20261016/glm53_production_gate.py --receipts benchmarks/joyner-glm53-20261016

On latest physical evidence (round a5c017204700): one valid pair,
0.400 tok/s fair share, no natural-language/logit parity, no 72h soak, no
fencing/rollback acceptance, therefore HOLD.

## Incremental release ladder

1. Physical benchmark-only experimentation on Joyner; no listener,
   autonomous admission, or global resource-policy change.
2. Read-only shadow service serving protected, test-only traffic;
   compare reference outputs and record latency/resource telemetry.
3. Authenticated restricted fleet canary with explicit fallback and
   independent health checks; never falsely report loaded/readiness.
4. 72h soak, injected failures, multiple users, process restarts,
   shutdown/readiness/rollback acceptance and operational monitoring.
5. Only after independent operator sign-off: bounded production rollout,
   service-level limits, documented rollback, persistent alerts.

Existing non-admitted R9700 canaries remain healthy and must not be
silently repurposed as GLM production. Every step preserves a HOLD until
its own acceptance evidence exists.
