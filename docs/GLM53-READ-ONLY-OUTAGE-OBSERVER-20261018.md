# GLM-5.3: out-of-band continuous availability incident observer

October 18, 2026. **READ-ONLY telemetry only**; GLM serving,
automatic route admission, credential renewal, and power operations remain
disabled. This x1-370 observer records remote outages independently of Joyner.

## Confirmed incident: independent Tailnet and LAN witnesses

- On x1-370, Joyner Tailscale 100.83.215.83 is offline, last seen ~1h ago;
  SSH and HTTP model service GET timed out.
- x1-370 DNS reported LAN hostname 192.168.1.79; independent xwing LAN
  observer could not establish TCP/22, TCP/18320 or ARP neighbor
  (INCOMPLETE/FAILED) for that address.
- x1 and xwing remained reachable. This is evidence of a host/link/address
  outage, not proof of physical power state, root cause or services.
- x1 fleet-node-health returns HTTP 200 healthy=true while no default
  inference route exists. Host liveness is NOT inference readiness.

## Implementation

- glm53_availability_probe.py: bounded nonredirecting GET-only transport
  and catalog check, without any route-admission authority.
- glm53_outage_observer.py: one-shot bounded sampler, Linux flock against
  concurrent runs, mode-0600 private JSONL evidence, atomic private state,
  refusal of symlink/hardlink/insecure paths and corrupt state, two 8MiB
  bounded journal segments (previous + current).
- Two consecutive failed samples escalate REMOTE_UNREACHABLE. HTTP 200
  or catalog recovery NEVER makes MODEL_READY. Three consecutive reachable
  samples merely establish transport observation.
- The systemd user service has a 15s timeout, MemoryMax=96M, CPUQuota=15%,
  read-only home/system except the dedicated private state directory.
  The user timer runs it every 60 seconds.
- The pinned source lives at a versioned home-local library path, not
  the moving Git worktree. Linux user lingering is already enabled on x1.

## Installation and rollback

Install on x1-370 only. Copy the two reviewed sources into the versioned
home library and verify SHA256; create the private state directory, install
named user-systemd units, daemon-reload, run a one-shot check; verify the
state and journal; then enable the timer and check a second observation.

Rollback stops and disables only colibri-glm53-availability-observer.timer.
Stop the named oneshot only if running; remove the two named unit files and
reload the user systemd manager. Preserve incident history. Do not touch
other inference units or restart Joyner without independent provenance.

## Remaining gate for actual inference service

Observation is not failover or GLM production serving. Production requires
verified loaded-generation proof and exact model/binary identity,
authenticated per-owner fences, admitted compatible fallback worker,
revocation and stale-completion protection, 72h serving/fault-injection
soak, bounded memory/SSD/GPU usage, and explicit recovery receipts.

**GLM production remains HOLD.**

## Actual x1-370 deployment on 2026-10-18 11:38–11:40 EDT

Linger=yes was already enabled for user scott, so the user-level timer is
eligible to survive logout and boot. Python source was copied to an
owner-private immutable versioned location:
  /home/scott/.local/lib/colibri-glm53-observer-20261018-v1/
The source SHA256 checks:
  glm53_availability_probe.py 36d41e645d525e050099f22e6d8a5ed9001fdc39318f14d72ba0ce0a7ccce7ef
  glm53_outage_observer.py c411f75c9af0f45480ba037be06e8dc874e18266aad24bed43baefd51fe7ee17

The initial systemd service failed before launching Python with
218/CAPABILITIES because x1's restricted user-manager cannot drop the
additional ProtectKernelModules/ProtectKernelTunables capabilities. Those
two optional controls were removed from the final unit (the service
already runs as an unprivileged user). The remaining NoNewPrivileges,
ProtectHome=read-only, ProtectSystem=strict, ReadWritePaths restriction,
CPUQuota=15%, MemoryMax=96MiB, TasksMax=16 and 15s execution timeout
remain enabled and were verified to execute successfully.

Two manual oneshots captured:
  sample 1: UNREACHABLE, REMOTE_UNREACHABLE_PENDING_CONFIRMATION
  sample 2: UNREACHABLE, REMOTE_UNREACHABLE (2 consecutive failures)
The named systemd timer was then enabled and started; an automatic third
observation appeared in the private journal, with consecutive_failures=3,
decision=HOLD, eligible_for_dispatch=false and failover_authorized=false.
No inference/model service, router, Tailscale config or Joyner process was
restarted or modified. Existing x1 fleet-node-health, fleet-node-reporter
and fleet-exec-relay remained active throughout.

Sample monitor operations:
  systemctl --user status colibri-glm53-availability-observer.timer
  systemctl --user status colibri-glm53-availability-observer.service
  cat ~/.local/state/colibri-glm53-observer/state.json
  tail -3 ~/.local/state/colibri-glm53-observer/observations.jsonl

The timer is a continuity probe, NOT a model-loader watchdog. It does
not prove true 24/7 serving, and it never performs automatic failover.
