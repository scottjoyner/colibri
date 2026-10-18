# GLM-5.3: host health is not model readiness

October 18, 2026. Status: **HOLD, probe-only**.

## Verified physical failure mode

x1-370's user-managed fleet-node-health.service is active, and its
local GET /health responds HTTP 200 with healthy=true. However,
GET /route provides default_route="" and homepage_route="".
This means the x1 host health check is green but does **not** prove that
a loaded inference model can serve a user.

Joyner's Tailnet address was offline in `tailscale status` on x1-370,
last seen approximately one hour earlier. SSH to Joyner timed out.
A second independent observer on xwing found Joyner reported offline,
and its direct GET /health to Joyner's model service timed out.
Neither observer proves whether the physical Joyner workstation has power
or whether its LAN-only services are healthy. No restart, power-cycle, SSH
mutation, or production route change was attempted.

## Implemented read-only differentiation

The source
`benchmarks/joyner-glm53-20261016/glm53_availability_probe.py`
makes only bounded unauthenticated GET calls. It distinguishes:

- UNREACHABLE: transport failed / timed out.
- HEALTH_FAILED: an HTTP error or malformed health response occurred;
  unlike UNREACHABLE, this may be an accessible-but-unhealthy service.
- HOST_HTTP_ONLY: a health endpoint responded but no valid catalog.
- MODEL_NOT_LISTED: model requested isn't in the /v1/models catalog.
- MODEL_CATALOG_UNVERIFIED: catalog lists the model; this **is not**
  attested LOADED state and proves no token generation.

The report **always** returns eligible_for_dispatch=false and
failover_authorized=false. It cannot issue inference requests, mutate
the gateway, load models, restart processes, or switch traffic.

17 unit controls exercise misleading HTTP 200, HTTP 503 versus TCP timeout,
wrong model, malformed data, oversized payload, invalid/protected URLs,
unexpected redirects and transport errors, and required fail-closed behavior.

Reproduction (from an authorized Tailnet observer):
python3 benchmarks/joyner-glm53-20261016/glm53_availability_probe.py --inference-base http://100.83.215.83:18320 --expected-model glm53

Latest recorded x1 observation:
- local_host_healthy=true
- local_gateway_advertised_route=false
- remote_state=UNREACHABLE
- model_loaded_verified=false
- decision=HOLD

The HTTP proxy and authentication edge are outside this component.
Production model-specific health needs a pinned source-model hash,
running binary identity, live successful bounded generation receipt,
per-user caller binding, and cryptographic/verifier-backed admission.

## Recovery / failover protocol before any 24/7 release

1. Obtain **independent** Joyner console/LAN/Tailnet witness. Check physical
   uptime, network interface, tailscaled status, user-service scopes,
   socket listeners, SSD health, R9700 driver reset messages, GPU memory
   state, and model checkpoint fingerprints. No blind reboot or generic
   process termination.
2. Exclude the failed node from new traffic using the admitted gateway's
   authenticated, generation-fenced registry; never infer availability
   from node-level HTTP 200 or stale model catalogs.
3. If a separately admitted compatible worker exists, perform a new
   owner-bound admission and verified loaded-generation health check.
   Reject duplicate/stale completions, revoked credentials, and cross-user
   execution receipt reuse. If no authorized fallback exists, show an
   explicit UNAVAILABLE state rather than pretending success.
4. Restore Joyner to an **isolated candidate** state. Run successful
   model generation proof, bounded canaries and rollback before admitting
   traffic. Startup readiness and liveness need independent definitions.
5. Prove 72h request soak, physical node-loss and recovery injection,
   two-user separation, stale-token fencing, and automatic rejection
   of unsupported or revoked model selections. Record recovery time and
   all incident evidence independently.

No production routing or GLM model admission was changed.
