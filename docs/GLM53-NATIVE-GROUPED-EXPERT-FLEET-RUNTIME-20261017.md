# GLM-5.3 native grouped expert I/O — candidate for the fleet runtime

**October 17, 2026 | Joyner AMD R9700 / Samsung 9100 PRO | Draft / NO-GO for serving**

Stacked on [Colibri PR #13](https://github.com/scottjoyner/colibri/pull/13).
This patch is **native GLM engine code**, unlike the PR #13 shadow-only reader.
The original production binary, both active Joyner llama.cpp endpoints, and
Fleet/AssistX admission policies have **not** been changed.

## Where it runs

Colibri's `c/glm53.c` streamed expert path uses `st_map_shard_range` when
`COLI_MAP_EXPERTS=1` and the mapping succeeds; **this opt-in mapping is
currently OFF by default**. Otherwise, streamed experts use one owner slot
and up to six buffered `pread` requests. Although `ref->contig` already
combines exactly adjacent logical-order tensors, real checkpoint tensors
are largely adjacent in **physical file order**, which often differs from
the logical gate/up/down order expected in memory. That is what the
new opt-in reader addresses.

### Runtime selection (default remains unchanged)

`GLM53_COALESCED_EXPERT_IO` is parsed at engine setup:

- Unset / `0` — existing behavior; no newly introduced read logic.
- `1` — physical-order grouped buffered `pread`; direct read into the
  pre-existing slot, logical pointers remapped. **No new 13.5 MiB scratch.**
- `vector` — Linux `preadv` per contiguous source extent, scattering
  directly into the original gate/up/down memory layout.
- `force` / `force-vector` — bypass the opt-in mmap fast path **only** for
  isolated A/B testing; never a default fleet setting.

Ordinary `1` and `vector` **preserve a successful mmap**. The entire
six-part read comes from one selected mirror replica, never a
cross-replica mixture. If the source is malformed, missing, overlapping,
out of bounds, or short, the native loader **fails closed**; it does not
silently use wrong weights or fall through to a different replica.

Grouping never spans a gap or shard boundary, and all six tensor sizes,
names, dtypes and positions remain authoritative from the checkpoint
loader. An expert remains 13.5 MiB in the 321B production checkpoint.
The pure plan allows one, two, four or six extents, not just the common
contiguous case.

## Physical tests on Joyner

The 32-event preregistered gating subset from PR #12 nominated **37
complete experts**. Every mode read the same private manifest of real
GLM-5.3 six-tensor source offsets; no checkpoint data was modified.

| Buffered native experiment | Six `pread` | Grouped packed `pread` | Scattered `preadv` |
|---|---:|---:|---:|
| Read calls / replay | 222 | 42 | 42 |
| p50 of first pass (ms) | 0.587 | 0.578 | 0.588 |
| p50 of reverse pass (ms) | 0.592 | 0.571 | 0.608 |
| All six-tensor payload digests | match | match | match |

The buffered read timing is **very sensitive to Linux page-cache warmth**.
Across this experiment p95/p99 vary substantially by order and first
touch. The earlier standalone O_DIRECT replay's ~12% median speedup
is **not transferable** to this buffered native path. Do not infer a
meaningful generated tokens/sec improvement from ~0.6 ms p50.

Full physical output receipts in
`benchmarks/joyner-glm53-20261016/hardware-20261017/native-runtime/`.
Per-expert source paths/offsets and the private training trace are not
committed. Run used `systemd-run --user --scope -p MemoryMax=384M
-p MemorySwapMax=0` with idle I/O priority. Both production endpoints
were HTTP 200 before/after, host memory PSI 0.

To reproduce this bounded **private** run on Joyner, copy the benchmark's
Python source dependencies from PR #13 into a private location, then run:

```bash
python3 export_native_grouped_manifest.py --rows 32 \
  --out /home/joyner/glm53-native-runtime-shadow-20261017/native_holdout.txt
systemd-run --user --scope -p MemoryMax=384M -p MemorySwapMax=0 \
  ionice -c3 nice -n 10 ./bench_glm53_native_shadow_vector_20261017 \
  /home/joyner/glm53-native-runtime-shadow-20261017/native_holdout.txt
```

The exporter uses exclusive 0600 creation, checks the pinned checkpoint
and retains all physical offsets locally. The C reader consumes six
canonical gate/up/down pieces, checks shard file bounds and fails closed
on short I/O or any mismatch across six sequential passes.

## Native correctness and Vulkan acceptance

- Standalone C plan fixture: one/two/four/six physical runs, permutation
  of logical pieces, genuine file reads, bounds/overlap/overflow denial.
- Linux scatter helper: `preadv` exact payloads, rejected EOF short reads,
  rejected negative offset and mismatched length.
- Both C test executables pass with UndefinedBehaviorSanitizer.
- Existing PR #12/#13 Python tests: 13 pass.
- Engine compiled successfully with `VK=0` and `VK=1` for baseline
  x86-64 and ran on **Joyner**, not only a simulation.
- Tiny independent GLM-5.3 resident-vs-int4 checkpoint pair:
  **eight distinct native loader configurations** compared exact
  `teacher_forcing` and greedy token sequences. **All equal** on CPU
  and R9700 Vulkan (16 comparison cases). Both `vector` and `1`
  actually executed native grouped reads with mapping disabled;
  with mapping enabled neither bypassed a successful mapping.
- The Vulkan test exercised the R9700 routed-expert tier, not merely
  GPU device enumeration. Warmup and benchmark runs were isolated;
  no production service was restarted.

## Fleet runtime gates (not waived)

1. Keep production unchanged: both active Joyner endpoints are
   authenticated, Tailnet-only llama.cpp services for other models.
   This PR does **not** create a GLM network listener, register a
   fleet model or authorize admission into Auto-Router/AssistX.
2. Full GLM inference previously caused **host memory PSI stalls**
   on Joyner's 14 GiB RAM. Headroom and dense offload need a separate
   bounded warmup/readiness proof, rollback and multi-tenant admission.
3. Run realistic held-out tasks with full native GLM
   exact logits/token parity plus first-token latency and sustained tok/s
   on an appropriate memory-safe node. Tiny-model and shadow parity
   cannot substitute for this.
4. Grouped native `preadv` is **not** yet joined to the PR #13
   imported-host Vulkan upload. Validate all six tensor offsets, GPU
   readback, memory/fence ownership and p99 in the same end-to-end test.
5. Add generation-fenced cache reservations, cancellation, eviction and
   deadline behavior. Gated routing predicts only 9.659% of selected
   expert labels in the PR #12 heldout data. Speculative reads remain
   disabled; cache accuracy versus extra bytes still needs improvement.
6. Preserve model source immutability, no implicit cold-start, no
   silent fallback to an unapproved model or CPU-only fleet claim.

**Rollback:** Leave `GLM53_COALESCED_EXPERT_IO` unset. On a later
authorized isolated GLM canary, omit the variable to restore native
loader behavior without reverting code. All deployment gates remain
**NO-GO** until the above evidence exists.
