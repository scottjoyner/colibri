# GLM-5.2 colibri as a local OpenAI-compatible provider

This repo already includes an OpenAI-compatible HTTP server in `c/openai_server.py`, exposed through `c/coli serve`. Use that before writing any separate shim.

This document records the currently verified provider profile for Scott's local GLM-5.2 colibri model on the Vulkan/iGPU branch. This model is for manual/offline emergency capability and supervised benchmarking only; do not wire it into unsupervised agent routing.

## Model and repo paths

```text
Repo:  /home/scott/git/colibri
Model: /media/scott/SSD_4TB/models-fast/GLM-5.2-colibri-int4-with-int8-mtp
Server: /home/scott/git/colibri/c/openai_server.py
CLI wrapper: /home/scott/git/colibri/c/coli
Engine: /home/scott/git/colibri/c/glm
```

## API surface

The server is OpenAI-compatible enough for standard local clients:

```text
GET  /health
GET  /v1/models
GET  /v1/models/{model_id}
POST /v1/chat/completions
POST /v1/completions
```

It supports streaming SSE, bearer auth, queueing, KV slots, and tool-call parsing. Model clients should use the `/v1` base URL.

## Build

```bash
cd /home/scott/git/colibri/c
make glm VULKAN=1
```

`VULKAN=1` links `backend_vulkan.c` behind the existing `COLI_CUDA` dispatch surface. Unsupported operations fall back to CPU.

## Stable provider launch profile

Use the stable Phase 2 Vulkan MoE path for provider mode:

```bash
cd /home/scott/git/colibri/c

export COLI_MODEL=/media/scott/SSD_4TB/models-fast/GLM-5.2-colibri-int4-with-int8-mtp
export COLI_MODEL_ID=glm-5.2-colibri-vulkan
export COLI_API_KEY=local-glm52

export COLI_CUDA=1
export COLI_NO_OMP_TUNE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json

# Do not enable these yet; Phase 3 attention is not usable.
unset CUDA_DENSE
unset COLI_CUDA_ATTN

# Strict-ish 25-30GB operating profile. `--ram 30` alone may auto-raise the cache cap.
export CAP_RAISE=0
export RAM_GB=30

# Simpler provider behavior until MTP is separately validated in server mode.
export MTP=0
export DRAFT=0

./coli serve \
  --model "$COLI_MODEL" \
  --engine ./glm \
  --host 127.0.0.1 \
  --port 8000 \
  --model-id "$COLI_MODEL_ID" \
  --api-key "$COLI_API_KEY" \
  --ram 30 \
  --ngen 512 \
  --kv-slots 1 \
  --max-queue 4 \
  --queue-timeout 600
```

### Why `CAP_RAISE=0` matters

In local testing, `--ram 30` by itself allowed the engine to auto-raise the expert cache cap to about a 50GB profile. For a real 25-30GB-ish target, set both:

```bash
export CAP_RAISE=0
export RAM_GB=30
```

and pass:

```bash
--ram 30
```

Verified strict profile:

```text
Projected peak: 23.1GB
Actual RSS:     17.76GB
Decode:         ~0.31 tok/s
Output:         coherent
```

Higher-RAM Phase 2 profile, where acceptable:

```text
RSS:    ~46GB
Decode: ~0.41-0.45 tok/s
```

## Smoke tests

Health:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Models:

```bash
AUTH_HEADER=$(printf 'Authorization: %s %s' Bearer "$COLI_API_KEY")
curl -s http://127.0.0.1:8000/v1/models \
  -H "$AUTH_HEADER" | python3 -m json.tool
```

Chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-colibri-vulkan",
    "messages": [
      {"role": "user", "content": "Explain the water cycle in two sentences."}
    ],
    "temperature": 0,
    "max_tokens": 96
  }' | python3 -m json.tool
```

Streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-colibri-vulkan",
    "stream": true,
    "messages": [{"role": "user", "content": "Give one practical use for a slow but high-quality local model."}],
    "temperature": 0,
    "max_tokens": 96
  }'
```

## Hermes provider config

Add this to `~/.hermes/config.yaml`:

```yaml
providers:
  glm52-local:
    base_url: http://127.0.0.1:8000/v1
    api_key: local-glm52
```

Then start a new Hermes session and use manually only when Scott explicitly wants to test or operate this model:

```bash
hermes chat --provider glm52-local --model glm-5.2-colibri-vulkan
```

Do not set it as an agent default. If Scott explicitly wants a temporary manual emergency/offline profile, it can be selected deliberately:

```bash
hermes config set model.provider glm52-local
hermes config set model.default glm-5.2-colibri-vulkan
```

Restart Hermes after provider changes, and revert defaults after the emergency/test window.

## Generic OpenAI-compatible client config

```text
Base URL: http://127.0.0.1:8000/v1
API key:  local-glm52
Model:    glm-5.2-colibri-vulkan
```

For clients that use OpenAI environment variables:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-glm52
```

Some clients use `OPENAI_API_BASE` instead:

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
```

## Suggested systemd user service

Create `~/.config/systemd/user/colibri-glm52.service`:

```ini
[Unit]
Description=Colibri GLM-5.2 OpenAI-compatible server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/scott/git/colibri/c

Environment=COLI_MODEL=/media/scott/SSD_4TB/models-fast/GLM-5.2-colibri-int4-with-int8-mtp
Environment=COLI_MODEL_ID=glm-5.2-colibri-vulkan
Environment=COLI_API_KEY=local-glm52
Environment=COLI_CUDA=1
Environment=COLI_NO_OMP_TUNE=1
Environment=VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
Environment=CAP_RAISE=0
Environment=RAM_GB=30
Environment=MTP=0
Environment=DRAFT=0

ExecStart=/home/scott/git/colibri/c/coli serve \
  --model /media/scott/SSD_4TB/models-fast/GLM-5.2-colibri-int4-with-int8-mtp \
  --engine /home/scott/git/colibri/c/glm \
  --host 127.0.0.1 \
  --port 8000 \
  --model-id glm-5.2-colibri-vulkan \
  --api-key local-glm52 \
  --ram 30 \
  --ngen 512 \
  --kv-slots 1 \
  --max-queue 4 \
  --queue-timeout 600

Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enable and verify:

```bash
systemctl --user daemon-reload
systemctl --user enable --now colibri-glm52.service
systemctl --user status colibri-glm52.service
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

## Fleet exposure

For a local Hermes session on the same machine, keep `--host 127.0.0.1`.

For fleet access over Tailscale, bind to a non-loopback address and keep bearer auth enabled:

```bash
--host 0.0.0.0 --port 8000
```

Then clients use the Tailscale host/IP:

```text
http://xwing.tailcb8954.ts.net:8000/v1
```

or:

```text
http://100.108.99.47:8000/v1
```

Start local-only first. Only expose over the tailnet after smoke tests pass.

## Routing guidance

Treat this endpoint as a slow, high-quality local fallback, not an interactive default.

Good uses:

- high-quality reasoning where latency is acceptable
- long-running local analysis that should avoid cloud providers
- tasks where smaller local models such as Orinth/Ornith are not enough
- background or queued work

Avoid for:

- low-latency chat
- rapid tool loops
- real-time coding loops
- tasks that require many short turns

Operationally, this model is closer to a dedicated-machine workload than a lightweight resident service. It can run on this device, but sustained use should be scheduled and isolated because it consumes memory bandwidth, SSD read bandwidth, and CPU/GPU resources for long periods.

## Known traps

- Do not enable `CUDA_DENSE=1` / `COLI_CUDA_ATTN=1` in provider mode yet. Attention hooks fire, but the current Phase 3 shader path is not correct or fast enough.
- `--ram 30` alone does not necessarily hold the process to a 30GB profile; use `CAP_RAISE=0 RAM_GB=30`.
- Keep `--kv-slots 1` until concurrency behavior is intentionally tested. This model is not suitable for many parallel requests on one workstation.
- Use `COLI_DEBUG=1` or `COLI_DEBUG=2` only while diagnosing prompt/output issues; debug logging can be noisy.


