# GLM 5.3 | Dual-node outage evidence and failover admission gate

October 18, 2026. GLM 321B production admission remains HOLD.

## Two physically separate, unprivileged observers are running

1. x1-370 periodically checks Joyner's Tailnet endpoint
   100.83.215.83:18320 and the local x1 fleet-health/route service.
   It has recorded more than 40 consecutive failed transport samples.
2. xwing periodically checks the distinct Joyner LAN endpoint
   192.168.1.79:18320. Its versioned source binaries match the x1
   observer SHA-256 hashes. It independently escalated after two failed
   samples; the xwing systemd user timer was enabled and verified to
   execute successfully at approximately 60s intervals.

xwing source paths:
  /home/scott/.local/lib/colibri-glm53-observer-20261018-v1/
xwing journal:
  /home/scott/.local/state/colibri-glm53-observer-xwing/
xwing unit files:
  colibri-glm53-availability-observer-xwing.service
  colibri-glm53-availability-observer-xwing.timer

Both are user-scoped, with no root credentials, MemoryMax=96 MiB,
CPUQuota=15%, a 15-second deadline, strict read-only system protection
except their private state directories, and bounded rotating journals.
x1/xwing user lingering is already enabled. Neither observer has a
model-generation route, task admission credential, wakeup/power command,
inference POST, or service-restart capability.

Important limits: x1 and xwing share fleet administration. They are
physically separate **availability observers**, NOT cryptographically
independent witnesses for rollback or split-brain authority. xwing's
local node-health HTTP service is absent, so its recorded
local_host_healthy=false is explicitly NOT a claim that xwing itself is
down; its systemd timer, SSH, and reporter were independently healthy.

## Read-only cross-node correlation

The CLI:
  python3 benchmarks/joyner-glm53-20261016/glm53_cross_witness.py
reads x1's 0600 state record, then obtains xwing's separate private
record over verified-host-key BatchMode SSH with a bounded stdout size
and execution timeout. Each state must be fresh (<=180s), strictly
read-only/non-admitting, coherent and non-replayed. The two observations
must be from the pinned distinct hosts and separate network paths,
and their timestamps differ by <=90s.

It returns one of:
- JOYNER_UNREACHABLE_BOTH_NETWORK_PATHS
- JOYNER_HEALTH_OR_TRANSPORT_FAILURE_BOTH_PATHS
- NETWORK_PATH_DISAGREEMENT
- MODEL_GENERATION_UNVERIFIED
- NO_QUORUM

Every classification preserves decision=HOLD, failover_authorized=false,
model_loaded_proven=false, eligible_for_dispatch=false.
It cannot authenticate an inference service, sign a receipt, redirect
clients, or choose/dispatch a different model. A stale/missing witness
produces NO_QUORUM, not a false positive recovery.

Physical read-only correlation returned
JOYNER_UNREACHABLE_BOTH_NETWORK_PATHS. The diagnostic does not establish
Joyner's power state or physical cause; LAN address assignment, switch
state, physical connectivity, and console require independent evidence.

## Newly observed fallback risk: xwing inference

While the monitor on xwing succeeded, its existing
fleet-node-health.service self-check reported **three failures**:
LM Studio native API :1234 was not serving, its reported model library
was empty, and the router lacked a loaded model there. The
lm-studio-api-start.service is inactive/dead but exited successfully
when last run; it is not a persistent runtime service, so one must not
label this a crash without further investigation. xwing separately has
a running CPU-only Ternary Bonsai worker, but that is NOT evidence of
an authenticated, compatible, admitted GLM fallback.

It would be unsafe to direct client traffic to xwing merely because
the machine is reachable. Before any production failover the gateway
must independently prove a compatible model is loaded, return successful
bound generation with exact binary/model identity, verify owner-specific
lease/fencing and revocation, and preserve failed requests with
idempotency and bounded retry. If no admitted fallback exists, clients
must receive explicit UNAVAILABLE; never forge 200 or auto-admit GLM.

## Acceptance before live 24/7 service

1. Diagnose Joyner from known-identity local console or out-of-band
   management. No blind power-cycle or reset of unrelated services.
2. Prove independent fallback worker and model-loaded generation; test
   2-user separation, stale completion rejection and revoke semantics.
3. Perform actual instrumented decode under >=3 isolated same-binary
   A/B pairs on varied prompts; prove parity and resource bounds.
4. Run 72-hour authenticated serving soak with host/GPU/SSD fault
   injection, loaded-model readiness, power/network loss recovery
   and reversible deployment.
5. Have operator review cryptographically verifiable evidence before
   approving any live router admission. Production currently HOLD.

## Rollback for new xwing witness

On xwing only:
  systemctl --user disable --now colibri-glm53-availability-observer-xwing.timer
  systemctl --user stop colibri-glm53-availability-observer-xwing.service
  systemctl --user daemon-reload

Remove only the two named xwing unit files after stopping them if
retiring the deployment. Preserve the private journal for incident
analysis; do not alter Joyner inference units or the fleet router.
