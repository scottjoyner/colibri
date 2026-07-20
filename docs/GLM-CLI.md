# `glm` — fleet front-door CLI

`glm` at the repo root routes by role to the model/endpoint defined in
`configs/colibri.toml`. It does not modify any default configuration.

```bash
glm ls                 # list configured roles (opencode, hermes, tiny-*)
glm health             # health of every configured endpoint (colibri-aware)
glm opencode           # serve the GLM-5.2 primary locally (via scripts/colibri-serve)
glm hermes             # show helper endpoint status / connection info
glm hermes chat "..."  # send a chat completion to the helper (remote)
glm tiny-<name>        # future small harness — same pattern as hermes
```

## Roles

- **opencode** → `[primary]` section: local GLM-5.2, served via
  `scripts/colibri-serve` with the HOT RAM + SSD KV tiers wired in.
- **hermes** → `[helper]` section (or `[hermes]`): the smaller fine-tuned LM on
  another device. `glm hermes` queries/health-checks it; `glm hermes chat`
  proxies a completion.
- **tiny-`<name>`** → `[tiny.<name>]` section: future small local harnesses
  (e.g. a traced-finetune eval loop). Same local-serve / remote-proxy pattern;
  add the section when the harness exists.

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
