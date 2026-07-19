# Investigation: engine produces degenerate text on GLM-5.2 (CPU and GPU)

Date: 2026-07-19 · Machine: xwing (Radeon 8050S) · Model: `/media/scott/data/GLM-5.2-colibri-int4-with-int8-mtp`

## Symptom

Every generation is incoherent garbage, on CPU **and** GPU, under greedy
(`TEMP=0 TOPK=1`) and sampling (`TEMP=1 TOPP=0.95`):

- "Explain the water cycle…" → `terms on1. the in3 llerves Ment else this evening;pageant…`
- "The capital of France is" → `resort, Previously1 A navig fir-0814 in a peace★Arendme1…`
- A correct GLM-5.2 would answer "Paris." / a coherent water-cycle paragraph.

This is **not** the Vulkan attention offload. The offload is numerically
correct (see "What is already fixed / proven" below). The incoherence happens
with `COLI_CUDA_ATTN=0` (pure CPU attention) too, so it is a base-engine /
model-loading problem.

## What is already fixed / proven (not the cause)

- `backend_vulkan.c` path-2 attention offload (3 shaders R/S/O) now matches the
  engine's own CPU reference to int4-float precision:
  - standalone `tests/test_attn_path2.c`: PASS (maxabs ≤ 0.125, maxrel ≤ 4e-3).
  - engine `COLI_ATTN_CMP`: GPU `out` vs CPU-ref maxdiff < 0.1 at all 78 prefill
    layers; decode ctx maxdiff < 0.015.
- Bug fixed: `g_attn_ds[2]` (o_proj descriptor set) was never allocated when the
  first attention call was `o=NULL` (absorb) → o_proj dispatch silently
  no-op'd → zero output. This was the old "layers 2-78 get zero → garbage"
  symptom for the GPU path. Fixed in `attn_ensure_resources`.

So the token divergence between GPU and CPU runs is just greedy-decode chaos
between two numerically-correct-but-not-bit-identical paths. The real blocker is
the base incoherence.

## Config sanity (ruled out dimension / hyperparameter mismatch)

Parsed from `config.json` and confirmed matching the engine's loaded `Cfg`:

| field | config | engine |
|---|---|---|
| num_attention_heads (H) | 64 | 64 |
| qk_nope_head_dim | 192 | qk_nope=192 |
| qk_rope_head_dim | 64 | qk_rope=64 |
| v_head_dim | 256 | v_head=256 |
| hidden_size | 6144 | D=6144 |
| num_hidden_layers | 78 | 78 |
| n_routed_experts / n_shared_experts | 256 / 1 | 256 / 1 |
| rope_theta | **8000000** | read via `rope_parameters.rope_theta` → 8M ✓ |
| rms_norm_eps | 1e-05 | read via `rms_norm_eps` → 1e-5 ✓ |

`json_get` is a flat one-level lookup (`json.h:153`) but `json_get(rp,"rope_theta")`
where `rp=json_get(r,"rope_parameters")` still resolves because `rope_theta` is a
direct key of the `rope_parameters` object. So **theta=8M and eps=1e-5 are
correctly loaded** — RoPE is not the cause.

## Prime suspects (in order to check)

1. **int4 conversion of this specific model dir is broken/corrupt.**
   The dir is `GLM-5.2-colibri-int4-with-int8-mtp` (note `-with-int8-mtp`). The
   coherent example in `glm52-hx370-2026-07-18.md` ("That is exactly right! The
   water cycle…") used a *different* model dir. Need to verify this dir converts
   and runs cleanly elsewhere, or re-convert. Check: does `.coli_usage` / a
   SCORE run give sane perplexity? (`SCORE` env exists in engine.)

2. **Tokenizer mismatch / wrong prompt tokens.** The engine tokenizes the prompt
   with `tokenizer.json` from the snapshot (`glm.c:4448 tok_encode`). If BOS /
   `[gMASK]` / `<sop>` handling is off, the model sees a wrong prefix. Note
   `glm.c:4346` prints a warning when "GLM config but tokenizer has no
   [gMASK]/<sop>: prefix OFF" — check engine stderr for that line.

3. **Engine MLA math bug orthogonal to the offload** (e.g. MoE routing,
   `routed_scaling_factor`, shared-expert, DSA indexer). Symptom: `experts
   loaded/token: 561.4 (per-layer 7.49)` with `TOPP=0.95` — a very high expert
   count; `first_k_dense_replace=3` dense layers. Decode expert routing may be
   selecting wrong experts.

4. **The `MTP ACTIVE (draft=0)` path** interfering with the main logits even
   when draft=0.

## How to verify quickly

- Run `SCORE` mode on a known continuation to get a perplexity number; a sane
  model gives low perplexity on in-distribution text. If perplexity is absurd,
  weights/tokenizer are the cause (1 or 2), not math.
- Compare a generation with `DRAFT=0` forced and with the MTP layer disabled, to
  rule out (4).
- Diff this model dir's conversion script output against the known-good hx370
  dir (same `convert_fp8_to_int4.py --ebits 4 --io-bits 8`).

## Files touched during this session (uncommitted)

- `c/backend_vulkan.c` — path-2 attention shaders + `attn_ensure_resources` ds[2]
  fix + `COLI_ATTN_DBG_O` gate (debug only).
- `c/glm.c` — added gated `COLI_ATTN_CMP` GPU-out vs CPU-ref comparison (debug
  only, off by default).
- `c/tests/test_attn_path2.c` — standalone numerical validation harness.
