# colibri GLM-5.2 endpoint — xwing startup (OpenAI-compatible)

Single-stream config is final for this APU (see `glm52-xwing-expert-io-2026-07-19.md`):
`COLI_CUDA_ATTN=1` only (iGPU matmul offload is a regression here), `RAM_GB=24`,
MTP_AUTO on by default. Concurrent `--kv-slots` raises per-user fairness on boxes
with bandwidth headroom but does NOT raise aggregate tok/s on xwing (measured:
~0.34 tok/s flat for N=1..3). Run with `--kv-slots 1` unless the other machine
has headroom.

## Prereqs
- Built engine: `cd c && make VULKAN=1 ARCH=native` (produces `c/glm`).
- AMD Vulkan ICD present: `/usr/share/vulkan/icd.d/radeon_icd.json`.
- Python deps for the server (Flask/werkzeug or stdlib — check `c/openai_server.py` imports).
- Model dir: `/media/scott/data/GLM-5.2-colibri-int4-with-int8-mtp`.

## Launch (port 1234, localhost)
```bash
cd /home/scott/git/colibri/c

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
export COLI_CUDA_ATTN=1 COLI_CUDA_ATTN_MIN_S=1
export RAM_GB=24
# optional tuning already defaulted: MTP_AUTO=1, PC_GB=2.5
# for multi-user fairness on a bandwidth-rich box, raise kv-slots (1..16);
# on xwing keep 1 (concurrency does not raise aggregate tok/s here)
export COLI_KV_SLOTS=1

python openai_server.py \
  --model /media/scott/data/GLM-5.2-colibri-int4-with-int8-mtp \
  --host 127.0.0.1 \
  --port 1234 \
  --model-id glm-5.2-colibri
```
The server binds port 1234 first, THEN loads the engine (so a bad/stale port fails fast).
It prints `OpenAI-compatible API listening on http://127.0.0.1:1234/v1` to stderr
once the engine is ready.

## Expose beyond localhost (only if needed)
Add `--api-key <secret>` (required by the server when host is not loopback) and bind
a routable host, e.g. `--host 0.0.0.0 --api-key "$COLI_API_KEY"`. Without an
api-key the server warns on non-loopback binds. CORS for browser clients:
`--cors-origin https://example.com` (repeatable; `*` for any).

## Talk to it (OpenAI client)
```bash
curl http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2-colibri","messages":[{"role":"user","content":"The capital of France is"}]}'
```
Python:
```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
print(c.chat.completions.create(model="glm-5.2-colibri",
      messages=[{"role":"user","content":"Hello"}]).choices[0].message.content)
```

## Notes / handoff
- This is the **xwing (Strix Halo APU)** config. The 96 GB HX370 box benefits from
  `COLI_FILECACHE=1` + `ulimit -l unlimited` (root) for mmap+mlock model cache,
  and can use higher `--kv-slots`. Those levers are not effective under the 24 GB
  mlock cap on xwing.
- Engine protocol (if debugging the subprocess directly): `run_serve_mux` via
  `SERVE=1 SERVE_BATCH=1 KV_SLOTS=N`, frames `SUBMIT <id> <slot> <bytes> <max_tokens> <temp> <top_p>\n<payload>\n`
  (payload byte-count excludes the trailing `\n`). Feed submits via file-redirect,
  not a pipe (pipe triggers `BAD_FRAME` on this build).
