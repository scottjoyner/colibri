# `glm` — fleet front-door CLI

`glm` at the repo root routes by role to the model/endpoint defined in
`configs/colibri.toml`. It does not modify any default configuration.

```bash
glm ls                 # list configured roles (opencode, hermes, tiny-*, harness)
glm health             # health of every configured endpoint (colibri-aware)
glm opencode           # serve the GLM-5.2 primary locally (via scripts/colibri-serve)
glm hermes config      # emit the Hermes cli-config.yaml [model] block
glm hermes check       # verify the hermes-adapter helper endpoint responds
glm hermes chat "..."  # send a chat completion to the helper (remote)
glm tiny-<name>        # future small harness — same pattern as hermes
glm harness "<task>"   # tier-3 orchestrator: fan subtasks out to subagents
glm proxy              # local OpenAI-compatible router: GLM <-> helper by token budget
glm route "<prompt>"   # dry-run: print which backend a request would hit
```

## Three tiers

- **Tier 1 — GLM-5.2 (primary)** `[primary]`: heavy reasoning/codegen on this
  box, with HOT RAM + SSD KV tiers. `glm opencode` serves it.
- **Tier 2 — hermes-adapter (helper)** `[helper]`/`[hermes]`: the smaller
  fine-tuned LM on another device. Hermes-agent speaks OpenAI-compatible via
  `provider:"custom"` + `base_url`; `glm hermes config` emits the exact
  `cli-config.yaml` `[model]` block to drop into Hermes so it routes through
  the helper. `glm hermes check` verifies the endpoint.
- **Tier 3 — self-contained harness** `[harness]` + `[tiny.<name>]`: a small
  orchestrator LM that decomposes a task into subtasks and fans them out to
  subagents (each a `tiny.<name>` endpoint) in parallel, then synthesizes.
  `glm harness "<task>"` runs the loop. `scripts/tiny-harness <name>
  [serve|train|eval]` is the per-subagent skeleton.

## Roles

- **opencode** → `[primary]` section: local GLM-5.2, served via
  `scripts/colibri-serve` with the HOT RAM + SSD KV tiers wired in.
- **hermes** → `[hermes]` (or `[helper]`): the smaller fine-tuned LM on
  another device. `glm hermes` queries/health-checks it; `glm hermes chat`
  proxies a completion; `glm hermes config` writes the Hermes adapter block.
- **tiny-`<name>`** → `[tiny.<name>]` section: future small local harnesses
  (e.g. a traced-finetune eval loop). Same local-serve / remote-proxy pattern;
  add the section when the harness exists. `scripts/tiny-harness <name>
  [serve|train|eval]` is the harness skeleton (serve wired; train/eval TODO).
- **harness** → `[harness]` section: orchestrator endpoint + subagent list.
  `glm harness` runs plan → parallel fan-out → synthesize.

## Proxy (token-budget routing)

`glm proxy` starts a local OpenAI-compatible front door on `[proxy].port`
(default 8200). Point opencode at it instead of GLM directly to get
transparent routing:

- Each request's token cost is estimated (~4 chars/token of prompt + requested
  `max_tokens`).
- If `estimate <= [proxy].route_max_tokens` → forwarded to the **helper**
  (cheap classify/route/format/scratch turns).
- Otherwise → forwarded to **GLM** (heavy reasoning/codegen).
- Upstream unreachable → clean `502` (no crash). Streaming passes through.

`glm route "your prompt"` prints the decision without serving — useful to
tune `route_max_tokens`.

## Config override

```bash
COLI_CONFIG=/path/to/colibri.toml glm opencode
```

## Health output

`glm health` verifies each endpoint is actually a colibri server (looks for
`kv_slots`/`scheduler` in `/health`) and prints the HOT tier status when
present:

```
  OK   opencode (primary)   http://127.0.0.1:8100/health  rss=46.8GB hot=3prompts/8.3MB hits=9
```
