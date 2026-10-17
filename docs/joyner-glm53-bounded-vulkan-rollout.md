# Joyner GLM-5.3 bounded dense loader — fork rollout (draft)

Owner repo: `scottjoyner/colibri`, branch `feature/joyner-glm53-bounded-vulkan-load-20261016`. This slice was branched from Joyner's **verified** Colibri evaluation commit `587712f`, not upstream `JustVugg/colibri`.

## Incident and evidence

On Joyner (R9700 32 GiB, system RAM ~14 GiB, ~6 GiB spare), Colibri's GLM-5.3 binary failed before Vulkan initialization on its fully SHA-256-verified 194.73 GB/62-shard model. An 11 GiB RAM-cap test was OOM-killed; a smaller 6 GiB RAM + 2 GiB swap test reached the full swap cap and was stopped. No generated tokens. Live inference remained healthy. GLM's `model_load_range()` loads resident dense weights prior to `coli_vk_init_env_tier()`, `g53c_start()`, and `COLI_VK_DENSE_HOST` releasing CPU copies.

The old `load_mat()` holds a full F32 matrix, `st_read_f32_cap()` temporarily holds another raw BF16/FP16 matrix, and `quantize_loaded()` allocates the final quantized output before releasing the F32 matrix. **This increases peak memory even above the accumulated resident weights.**

## Slice 1: bounded matrix staging (implemented, opt-in)

- `GLM53_BOUNDED_DENSE=1` enables a fork-only alternative for GLM-5.3 resident matrices with `GLM53_BITS=4` or `8`.
- Reads at most about 2 MiB F32 worth of complete matrix rows at once. The corresponding raw BF16/FP16/F32 buffer is similarly bounded. Nothing is modified in the source checkpoint.
- Uses the exact legacy `quantize_i4_grouped()` and `quantize_rows()` row quantizers; no change to quantized values, layout, scales, model shapes, tokenization, expert routing, MTP, DSA or KDA.
- Checkpoints with prepacked U8 routed experts continue on the original validated path.
- The `GLM53_BITS=32` path stays on the original loader. Default is unchanged when the opt-in flag is unset.
- A hostile header/type/short-read must be rejected before publishing any quantized result.

**Scope warning:** this reduces **transient** host RAM usage. Persistent CPU-resident weight arrays still accumulate until Vulkan starts. This is NOT a full 14 GiB RAM feasibility fix. It must not be promoted or used as a production admission claim.

## Slice 2: file-backed resident quantized matrices (experimental, separate opt-in)

- With both `GLM53_BOUNDED_DENSE=1` and `GLM53_DENSE_SPILL_DIR=/path/on/local/ssd`, store each matrix's packed quantized bytes and row/group scales in a private 0600 file-backed `MAP_SHARED` mapping. The name is immediately unlinked, the pages are synced and advised reclaimable, and the device may read those pages until GPU upload completes.
- `GLM53_DENSE_SPILL_DIR` must point to an **existing, writable, disk-backed** directory. Do not use /tmp on tmpfs, RAM disk or the authoritative checkpoint location. Suggested Joyner isolation: `/srv/joyner-gen5/colibri-fork-dense-stage`. The benchmark runner must verify this mount's UUID before launching.
- An upload failure leaves the mapping readable by the CPU; successful device-only placement uses exactly one `munmap` and clears pointers. Device-loss reload transfers the mapping ownership into the resident matrix, and final model release also calls `munmap`. Unlinked files do not persist after process termination.
- This is a local **experiment**, not a general disk-spill feature ready to ship. Invalid paths, truncated weights and failed writes/sync must fail closed.
- Existing no-flag behavior is still malloc-backed and unchanged. The optimization is never enabled by default. The expected benefit is lower anonymous RSS during pre-Vulkan initialization; the OS still needs enough free RAM for embedding/vision/KDA state and GPU allocations.

## Test gates

- Byte-identical matrix results for BF16, FP16 and F32 source rows, gs64-int4 and int8 fallback, including windows crossing a chunk boundary; repeat for heap and disk-backed mmap allocations.
- Negative cases for invalid tensor dtype, wrong declared length, and a truncated file in both modes, plus no orphaned named files after mapped runs.
- Build default/opt-in. Next: run fixture-level C inference against the exact pretrained reference and the unmodified Joyner binary with `GLM53_BITS=4` and `8`.
- Run bounded host-memory profiling; then implement **incremental Vulkan layer placement and release** as a separate slice with fail-closed fallback/rollback, numerical oracle and full checkpoint canary.

## Reference discipline from k2-mova

`scottjoyner/k2-mova` contains model/oracle and dtype-consistency tests from Strix Halo AIE/ONNX investigation. A successful compilation does **not** establish numerical correctness. Reuse its verification discipline (multiple prompts, absolute/relative logit deltas, negative branches and dtype invariants) rather than assuming AIE2 kernel code is transferable to Vulkan.

## Staged rollout

1. This branch in the owned fork only. No direct upstream changes.
2. Numerical unit cases on x1-370, build/fixture proof on isolated Joyner checkout. No production route/service restart.
3. Real full-model memory experiment only after a **file-backed or incremental offload** mechanism caps persistent RSS; do not repeat the known OOM.
4. Require benchmark receipts for peak RSS, VRAM, first-token latency, cached vs uncached expert bytes, short/long-prompt tokens/sec and source SHA256.
5. Draft PR and explicit opt-in; later operator-approved canary. Production admission HOLD until these tests pass.

## Measured hardware gates — Joyner, October 16 at 23:19–23:26 EDT

**Numerical validation:** fork Vulkan build passed. The 24 isolated tensor cases passed (18 byte-identical int4/int8 conversions across BF16/F16/F32, 6 invalid-truncation rejections, heap and disk-backed modes). Against the unchanged Joyner engine, 3 synthetic fixtures at bits 4 and 8 showed **six identical teacher-forcing, greedy-token and full-logit sequences** in both heap-backed and Gen5 mmap modes. Files under `benchmarks/joyner-glm53-20261016/`.

**Real full checkpoint success:** `GLM53_BOUNDED_DENSE=1`, `GLM53_DENSE_SPILL_DIR=/srv/joyner-gen5/colibri-fork-dense-stage` on the UUID-pinned 9100 PRO, `COLI_VULKAN=1`, `COLI_VK_DEV=1`, `COLI_VK_CHAIN=1`, `COLI_VK_DENSE_HOST=1`, `COLI_VK_TIER_GB=2`, `GLM53_EXPERT_GB=0.05`, `GLM53_BITS=4`, and 6 GiB hard RAM cap, swap disabled. The model loaded 45/45 layers on GFX1201 and **generated two token IDs**, `154822 154822`, in 3.3 s (**0.604 tok/s**) after a 94.3 s single-token prefill and ~30 s load. The test had significant reclaim throttling under initial MemoryHigh=5 GiB; raising only the *soft* cap to 5800M allowed completion, with no swap or OOM. The most meaningful result is first full-model execution with <6 GiB process RAM, **not production usable decode**.

**Cache miss evidence:** 994 expert cache misses, ~14.1 GB expert bytes read, 1 CPU RAM hit, 13/1008 routed expert assignments on the device and 48 promoted experts (576 MiB uploads), even though all 45 dense chain layers were GPU resident. Device expert tier target 2 GiB; GPU execution is still bottlenecked by cold expert I/O and repeated read patterns. The log's mixed accounting should be investigated rather than naively converted to an expert hit rate.

**Natural prompt NO-GO:** A separate 17-input-token text prompt, `Hello. Respond with a brief greeting.`, with `--greedy 8` and the same bounded memory configuration exited from `timeout 230s` with status **124** before reaching a completed generated response. Peak cgroup memory approximately 6.18 decimal GB (<6 GiB hard), no swap. **Natural-language correctness, sustained inference and quality remain unverified.** Do not report this as a usable conversational endpoint.

**Safety:** Both production ports 18320/18321 stayed HTTP 200. No runtime/model admission, upstream repo, or production server settings were changed. The private staging directory was empty of named files at the end of the successful test. These runs are not acceptance for production promotion.

**Next required validation:** investigate cold expert I/O and tier policy with bounded real-token instrumentation; independently confirm short natural-language outputs, tokenizer stop handling, and low-RAM reload/drop behavior before enlarging context or moving to multiuser benchmarks. A separate more-RAM host such as Strix Halo can distinguish the loader's memory benefit from an I/O bottleneck.
