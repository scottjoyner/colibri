# GLM-5.3 — predictive expert residency: evidence-first rollout

Owned implementation: `scottjoyner/colibri` fork; stacked on the low-memory loader draft PR. All inference path changes OFF by default. **This slice collects labels and permits offline predictors; it does not prefetch anything.**

## Research / interfaces

Three different problems must be kept separate:

1. **Next expert prediction, not next token:** train on the *previous* layer's router IDs + a 16-dimensional signed activation sketch and known cache residency; predict the next layer's top-8 experts. Prefetch during attention/other compute, but never change Colibri's authoritative native top-K router. See *Pre-Attention Expert Prediction and Prefetching* (arXiv:2511.10676), *SpecMD: A Comprehensive Study On Speculative Expert Prefetching* (ICML 2026, arXiv:2602.03921), *MoE-SpeQ* (arXiv:2511.14102), and *ExpertFlow* (arXiv:2410.17954).
2. **Token speculation:** DFlash's block-diffusion drafter proposes several future tokens and the **target model verifies** them. Official DFlash2 repo currently lists Muse-Glimmer-30B and Qwen3.8-27B, **not GLM-5.3**. No drop-in GLM-5.3 DFlash2 weights have been demonstrated; do not claim DFlash speedups here. [DFlash paper](https://proceedings.mlr.press/v306/chen26bv.html) / [DFlash repo](https://github.com/z-lab/dflash).
3. **Tier placement:** Gen5 NVMe -> OS page cache -> explicit host expert slot -> R9700 VRAM. A missing RAM/VRAM slot may still be in the Linux page cache. Distinguish physical SSD bytes, host memcpy, VRAM uploads, effective first-token time, cache collisions and wasted speculation. Use explicit per-tier occupancy/bandwidth caps, async cancellation/fencing and inference-session ownership.

The DFlash2 architecture is valuable as an *asynchronous lookahead pattern*, but the first safe prototype predicts expert IDs via an auxiliary head while the *native model* determines all selected experts and output logits.

## Capture ABI

Set `GLM53_EXPERT_TRACE=/absolute/path/new-unique-file.jsonl` only for private, opt-in model benchmark runs. It opens the path exclusively (no overwrite), chmods 0600, and caps rows at 100000. Do not trace user chats. One JSONL event per model router invocation and row:
- v=1; monotonically increasing `call`; per-batch `row`, `layer`, `n_experts=288`, `topk=8`.
- `ids`, `gates` are actual native top-8 post-normalization expert selections; never predicted.
- `resident`: 2 = GPU expert tier; 1 = host slot; 0 = no known application-level resident copy (not necessarily a physical SSD miss). These states are captured **before** real expert loading, with no LRU/access-clock mutation.
- `sketch`: 16 fixed, signed bucket projections of the router input. This feature can still leak information; traces must remain private. No prompt strings, raw token IDs, complete hidden states, or private output text.

The dataset's `call` field increases once per MoE invocation. Replay only connects `(call-1,row,layer-1)` to `(call,row,layer)` in the same model run, preventing lookahead to its target label. **Run files are the data partition boundary**; never mix the same run's rows between train and evaluation.

`benchmarks/joyner-glm53-20261016/replay_expert_prefetch.py` measures held-out native expert recall and upper bounds on avoided misses and wasted expert bytes for layer-frequency / previous-token / cross-layer predictors. The estimates are **counterfactual**, not observed speedups, and MUST NEVER be reported as achieved throughput.

`train_expert_attention.py` is a small optional PyTorch 4-tap, four-head cross-layer-attention model (learned layer and expert embeddings + signed sketch input -> 288 native expert logits). It requires at least **three distinct training traces**, 2000 **cross-layer pairs**, and a fully separate held-out trace to train; otherwise it reports NO-GO. It only writes offline model weights and metrics; it cannot trigger reads or affect the router. PyTorch must be installed by the experiment environment, not production services.

## Hardware qualification sequence

1. Test a no-trace vs trace **logit/token identity** A/B against both heap and NVMe-backed fixture modes. Validate JSON event counts/shape, 0600 ownership and existing-file refusal. Keep original `ROUTE_TRACE` format untouched.
2. Instrument real verified 62-shard GLM with `GLM53_EXPERT_TRACE`, memory cap 6 GiB, no swap, GPU tier 2 GiB, and pinned SSD mount UUID. Benchmark only explicit synthetic benchmark prompts. Maintain existing 18320/18321 production services untouched.
3. Capture >=4 independent runs of representative workloads (short greetings, code edits, retrieved text, repeat/prefix) and strictly held-out evaluation; with current cold expert latency a staged workload set is needed. Count valid cross-layer events before training.
4. Compare oracle, frequency, LRU, Least-Stale-like, Markov, and attention recall@8/16/24/32, per-layer mean and tails, cold expert bytes, p95, TTFT; measure energy from Strix Halo and R9700 where available. The replay does not count OS page cache hits as SSD reads.
5. Only if predictive accuracy is validated, implement a **read-only shadow planner** that outputs candidate prefetch IDs and predicted transfer bytes; do not perform transfers.
6. Only after measured head gains, introduce guarded async expert prefetch with byte quota, source revision and quant pinning, per-session cancellation and rollback. Native expert loads always win; no speculative routing changes.
7. Separately qualify true DFlash2-style token proposals with proper GLM drafter/verify support; measure token acceptance, verifier GPU cost, KV consistency, and model quality.

## Existing measured baseline (not a predictor benchmark)

Joyner's real GLM two-token smoke achieved only **0.604 tok/s** after 94.3s prefill, with 994 expert misses and ~14.1 GB loads. A 17-token natural prompt timed out at 230s. Optimizing prediction may not solve the synchronous prefill or RAM pressure alone: measure before attributing any acceleration to it.

Rollback: do not set the trace env and the only hot-path cost is a never-open branch; the native model/router/cache remain functionally unchanged. No AssistX deployment or upstream changes.
