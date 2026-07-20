# Fleet Status — local colibri inference fleet

> **Status: PRE-PRODUCTION. DISARMED by default.** Nothing binds a port, launches
> the 744B engine, fans out via the harness, sends chat traffic, or fine-tunes
> until the operator explicitly arms the fleet (`export COLI_LIVE=1`). This file
> records what is wired, what is verified *in isolation*, what is blocked, and the
> pre-flight / go-live / rollback procedure.

## TL;DR

```bash
glm doctor                 # read-only validation of the whole fleet config
# when ready to actually run:
export COLI_LIVE=1         # ARM (only in the shell you launch from)
glm opencode               # serve GLM-5.2 primary (HOT RAM + SSD KV tiers)
glm proxy                  # token-budget router (port 8200)
glm harness "<task>"       # tier-3 orchestrator -> subagents
# tiny harness (per-subagent fine-tune):
tiny-harness coder train --plan      # show resolved plan (safe, no COLI_LIVE needed)
tiny-harness coder train --run       # REAL QLoRA (needs COLI_LIVE=1)
tiny-harness coder serve             # serve the fine-tuned checkpoint
unset COLI_LIVE              # DISARM again
```

## What is wired

| Component | Entry point | Notes |
|-----------|-------------|-------|
| GLM-5.2 primary + HOT RAM/SSD KV | `glm opencode` / `scripts/colibri-serve` | KVDB self-test passes; prefill becomes a memcpy on prefix hit |
| Helper adapter (Hermes) | `glm hermes config\|check\|chat` | emits Hermes `cli-config.yaml` `[model]` block; `--apply` writes it |
| Token-budget proxy | `glm proxy` | routes ≤ `route_max_tokens` to helper, else GLM; streaming passthrough |
| Tier-3 harness | `glm harness "<task>"` | orchestrator plans subtasks → fans out to `tiny.<name>` in parallel → synthesizes |
| Subagent fine-tune | `scripts/tiny-harness <name> train` | drives the `auto-finetune` pipeline (extract→clean→format→train) on opencode traces |

## What is verified (in isolation)

- **KV cache tiers**: `KVDB_SELFTEST=1` → `PASS: HOT RAM served full prefill (P=19) in 0.19ms (FORWARD COMPLETO SALTATO)`.
- **Proxy routing + `/health`**: verified routing both directions and 502 on upstream-down (unit + manual).
- **`glm hermes config` / `glm harness` graceful failure**: verified output when endpoints are down.
- **`glm doctor`**: validates config, ports, model paths, endpoint reachability, training prerequisites (read-only).
- **`tiny-harness train`**: `--plan` resolves; shim runs the real `auto-finetune` `train.main` dry-run against the built dataset (override + tokenization OK).
- **Tests**: `c/tests` — 70 passed (incl. `_stats` kvdb-trailer parse).

## What is blocked / NOT yet observed live

- **Live GLM↔helper round-trip** and **live `kv_cache_hot` via `/health`** could not be
  observed end-to-end **because this tool kills background processes after ~120 s** and the
  744B engine alone takes ~90–120 s just to load. All code paths above are verified in
  isolation; the live run needs a stable process (a `systemctl --user` unit, or a detached
  `nohup` in your own shell) outside this tool.
- **Real QLoRA training** is a multi-hour GPU job; not executed here. `train --run` is
  gated behind `COLI_LIVE=1` and additionally guarded (see Hardening).

## Hardening already in place

### Fleet-wide arming gate (`COLI_LIVE`)
All live actions refuse to run unless `COLI_LIVE=1` in the environment:
`glm proxy`, `glm opencode`/`tiny-<name> serve`, `glm harness`, `glm hermes chat`,
and `tiny-harness <name> serve` / `train --run` / `eval --live`. Read-only commands
(`ls`, `health`, `route`, `doctor`, `hermes config|check`, `tiny-harness train --plan`)
are always allowed. `glm arm` / `glm disarm` print the exact `export`/`unset`.

### `glm doctor` pre-flight (read-only)
Checks: config sections present; port collisions + availability; local model paths exist;
remote helper/hermes endpoints reachable; per-subagent training prerequisites
(`train_repo/src`, `train_base` exists, `train_output` parent writable, valid
`train_source`). Prints PASS/WARN/FAIL; exits non-zero if any FAIL.

### Proxy hardening
- **Request size cap** (`proxy.max_body_bytes`, default 8 MiB) → `413` on oversized bodies.
- **Circuit breaker** (`proxy.breaker_failures` / `breaker_cooldown_s`) → after N
  consecutive upstream failures the upstream returns `503` for the cooldown window instead
  of hanging every request for the 600 s timeout.

### Hermes hardening
- `glm hermes config` redacts the `api_key` by default (`--show-key` to reveal;
  `--apply` writes the file with a `.bak` backup; `--out=PATH` chooses destination).
- `glm hermes check` verifies `/health` **and** that the configured `model_id` is actually
  advertised by `/v1/models` (warns if not), so a half-configured adapter is caught early.

### tiny-harness training hardening
- **Armed gate** on `serve` and `train --run` / `eval --live`.
- **Base-model existence check**: refuses if `train_base`/`model` path is missing.
- **Lockfile** (`<output_dir>.lock`): refuses to start if another run for the same
  subagent is active (PID-alive check).
- **No-clobber**: refuses to overwrite existing checkpoints unless `--overwrite`.
- **Run manifest**: writes `<output_dir>/run-manifest.json` (resolved args, base, output,
  source, pid, timestamp) for an audit trail of every run.
- **Empty-dataset abort**: after prep, if `format` produced 0 trainable examples, the run
  aborts instead of burning GPU time on an empty set.

## Go-live checklist

1. `glm doctor` → resolve all FAIL (and any WARN you care about).
2. Confirm model/endpoint paths in `configs/colibri.toml` match this machine.
3. For any `tiny.<name>` you will train: confirm `train_base` exists, `train_output`
   parent is writable, and the opencode trace DBs are mounted.
4. `export COLI_LIVE=1`.
5. Start the primary: `glm opencode` (background / systemd).
6. Start the proxy: `glm proxy` (background / systemd).
7. Optional: `glm hermes check`, then point Hermes at `glm hermes config --apply`.
8. Optional: `tiny-harness <name> train --run` on the GPU box, then `tiny-harness <name> serve`.
9. Smoke test: `glm route "classify this"` → should route to helper; a long prompt → GLM.

## Rollback

- `unset COLI_LIVE` stops any *new* live action from starting (in-flight servers keep
  running until you kill them).
- Kill the server/proxy PIDs; the KV cache (`.coli_kvdb/`) and dataset staging dirs are
  side-effect-only and safe to keep or delete.
- `tiny-harness <name> train --overwrite` replaces a checkpoint dir on the next run; the
  previous run's `run-manifest.json` is overwritten, not accumulated — copy it first if you
  need history.
- Hermes: `glm hermes config --apply` keeps a `.bak`; restore it to revert.
