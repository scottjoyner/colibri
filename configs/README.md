# colibri local fleet — launchers & configs

These files wire the GLM-5.2 primary (this HX370 box) with the HOT RAM + SSD
KV cache tiers, and declare the helper-model endpoint(s) for the smaller LM
fine-tuned on opencode traces (run on another device). They do **not** modify
any default config.

## Files

- `colibri.toml` — central profile: primary model, cache tiers, prompt-prime
  dir, and helper endpoint. Edit this, then launch.
- `scripts/colibri-serve` — reads `colibri.toml`, exports the engine env
  (`COLI_KVDB` / `COLI_KVHOT_GB` / `COLI_KVDB_NORMALIZE` / `COLI_KVDB_PRIME`)
  and execs `coli serve`. Accepts a config path arg or `$COLI_CONFIG`.
- `opencode.colibri.jsonc` — opencode provider config: GLM as primary, helper
  on the other device. Copy/merge into `~/.config/opencode/opencode.jsonc`.
- `systemd/colibri-glm52.service` — user service launching via the script.

## Launch (this box)

```bash
# one-shot
scripts/colibri-serve configs/colibri.toml

# or as a service
mkdir -p ~/.config/systemd/user
cp configs/systemd/colibri-glm52.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now colibri-glm52.service
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

The `/health` response now includes `kv_cache_hot{prompts,size_mb,hits,
promoted_from_ssd}` so you can confirm the HOT tier is populated and being hit.

## Prewarming static system prompts (free first request)

Drop Hermes/opencode system-prompt files (`.json/.md/.txt/.yaml/.yml`) into a
directory and point `cache.prime_dir` at it. At startup the engine prefills
each once into HOT+SSD, so the first request of every session that sends the
same prompt is a free memcpy (no 70s cold prefill). The prompt text is wrapped
with the same GLM chat template the server uses, so the cache sha matches.

## Helper model (other device)

Run the fine-tuned helper with its own `coli serve` (or any OpenAI-compatible
server) on the helper device, expose it at `helper.base_url`, and point
opencode at it via `opencode.colibri.jsonc`. Route cheap/high-frequency turns
(classify, route, tool-shaping, scratch) to the helper; GLM-5.2 handles heavy
reasoning/codegen. Keeping the helper off-box leaves this machine's 96 GB for
the 744B expert cache, raising GLM decode throughput.

For parallel subagents each on a smaller local model, give each its own
`coli serve` instance (own port / device) and add a provider entry per agent
in opencode — the same pattern as `hermes-helper` above.
