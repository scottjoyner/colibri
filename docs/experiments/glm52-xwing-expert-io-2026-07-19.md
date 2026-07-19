# Expert I/O investigation — xwing (Radeon 8050S, 20 GB cap)

Date: 2026-07-19 · Machine: **xwing** (AMD Ryzen AI MAX PRO 390 / Radeon 8050S,
Strix Halo) · Model: `/media/scott/data/GLM-5.2-colibri-int4-with-int8-mtp`
(single 2.67 GB `out-00000.safetensors`) · RAM budget: **RAM_GB=20** · NVMe
`/dev/nvme0n1p3` (~27 GB/s sequential, file fully fits in OS page cache).

This note records the expert-load bottleneck analysis on xwing and the three
changes shipped in commit `457a536`. It complements
`glm52-hx370-2026-07-18.md` (the *other* HX 370, 96 GB, different
144-shard snapshot) and `glm52-incoherent-text-investigation.md`.

## TL;DR

The reported bottleneck — `PROFILE: expert-disk 0.000s service / 21.6s wait` —
is **not disk bandwidth**. `service=0.000s` only times the async *dispatch*
enqueue (by design, it is instant). The 21.6s is the forward thread spinning
in `pipe_wait()` waiting for PIPE I/O workers. Measured disk reads are 27 GB/s
and the whole 2.67 GB model file stays resident in the page cache, so every
expert load is a cached `pread` + copy. The cost is **CPU-side per-expert work
and OMP-matmul ↔ I/O-worker contention**, not bytes moved off NVMe.

End-to-end on xwing (attention offload `COLI_CUDA_ATTN=1`, RAM_GB=20,
`PROMPT="The capital of France is"`, NGEN=8):

| config | tok/s | experts/token | expert-disk wait | notes |
|---|---|---|---|---|
| baseline (default, MTP on, draft=3) | 0.27 | 1670 | 21.6s | MTP loads 3× redundant experts |
| `MTP=0` | 0.40 | 525 | 15.4s | correct ~topk×layers load count |
| `MTP=0 PIPE_WORKERS=16` | 0.41 | 525 | 14.9s | workers not the limiter |
| `COLI_MMAP=1` | 0.09 | 1595 | 36.6s | mmap+4KB pre-touch is 3× *worse* |
| **optimized default (MTP_AUTO on)** | **0.38** | 679 | 16.2s | +41% vs baseline, automatic |

## What was actually measured

1. **Disk is not the bottleneck.** `dd` of the 2.67 GB file: 26.8 GB/s. A
   synthetic random 6 MB × 570-tensor `pread` loop (simulating one token's
   expert loads): 27.6 GB/s, 0.12s pure-disk. So 570 cached expert reads
   cost ~0.1s of I/O — the 21.6s wait is elsewhere.
2. **Warm cache = cold cache.** A second identical run (file already in page
   cache) showed 29.7s wait vs 21.6s cold — *no improvement*. Confirms the
   file is always cached; the wait is CPU, not disk.
3. **`COLI_MMAP=1` is harmful here.** The mmap path (glm.c:1679) does a
   4 KB page pre-touch per tensor (glm.c:1702-1719) = ~855k page faults/token.
   Under 20 GB RAM the 12 GB dense + 6 GB reserve leave no room, so most
   faults go hard → 0.09 tok/s (3× slower than the pread+slab default).
   **Keep the default pread/slab path.**
4. **MTP is a net loss at this speed.** With ~25% draft acceptance, MTP loads
   ~3× experts (1670 vs 525/token) for draft positions that get rejected.
   Turning it off drops the load count to the correct 525 and gives +48%.
5. **Worker count is not the limiter.** `PIPE_WORKERS=16` vs default 8 gave
   0.41 vs 0.40 tok/s — negligible. The forward thread serializes: it must
   `pipe_wait(qof[j])` for each expert *in order* before matmuling it, while the
   matmul's OpenMP team (many cores) starves the 8 I/O workers of CPU. The
   pipeline stages effectively run at ~1×.

## Changes shipped (commit `457a536`)

### 1. `MTP_AUTO` (default ON) — the real win (+41%)
In `spec_decode`, sample per-forward expert-disk wait vs decode wall-time. Over a
~3s window, if wait exceeds **50%** of decode, latch `m->has_mtp=0` for the
rest of the run and print:
```
[MTP_AUTO] expert-disk wait 71% of decode (5.54s/7.81s): disabling MTP
           drafting for the rest of the run (MTP_AUTO=0 to keep)
```
Opt-outs: `MTP_AUTO=0` keeps MTP on; explicit `MTP=0` still wins at startup.
This auto-recovers the +48% `MTP=0` result with no user action, and will
*keep* MTP on machines where acceptance is high enough that drafting pays off
(e.g. the 96 GB HX 370 when experts are mostly resident — see below).

### 2. Pre-allocated `ws[64]` dispatch slabs
`expert_load` (glm.c:1727) re-allocates a slab only if `slab==NULL` or
`wtot+8192 > slab_cap`. Every expert has identical dims, so `wtot` is constant
— after the first load the slab persists and the LRU swap (glm.c:3200) only
moves pointers, never frees. The change **pre-allocates the shared `ws[64]`
slabs once at `model_init`** (sized from `expert_bytes_probe`), removing the
first-dispatch `posix_memalign` and guaranteeing persistence. Neutral on timing
(the old path already avoided realloc), but makes residency robust and removes a
per-dispatch branch.

### 3. `PC_GB` tunable for the page-cache reserve
The RAM budget reserves a fixed 2.5 GB page-cache margin (`pc_b`,
glm.c:5757) to keep buffered `pread` throughput up. At RAM_GB=20 this margin
plus the 12 GB dense forces `cap=1` expert LRU residency, so ~every token
reloads all experts. `PC_GB=<n>` lets the user trade that margin for more
expert cache. Tested 2.5 → 1.0: cap rose 1→2 but tok/s was flat (0.34-0.37)
and RSS rose toward the wall — at 20 GB the 12 GB dense dominates, so this knob
is **ineffective on xwing** but is a real lever on bigger-RAM Strix Halo boxes
(see cross-box note).

## Where the time goes (optimized default, decode 8 tokens)
```
PROFILE: expert-disk 0.000s service / 16.173s wait | expert-matmul 2.389s
          | attention 1.419s (incl kvb 0.000s) | lm_head 0.000s | other 0.820s
```
- expert-matmul (pure CPU int4): **2.4s** — compute is cheap.
- attention (Vulkan offload): **1.4s**.
- expert-disk **wait: 16.2s** — the floor. Forward thread blocks on ordered
  `pipe_wait` while OMP matmul starves the I/O workers.

## Deeper architectural change (next step)
The 16s wait is **forward-thread serialization + CPU contention**, not I/O. The
current PIPE design dispatches a whole block's misses then waits for each expert
*in token order* before matmuling it — so even with 8 workers the pipeline runs
at ~1× and the matmul's OMP team starves them. Real levers:

- **Decouple matmul from the wait:** a consumer thread that matmuls experts the
  moment their `ready[q]` flips, while the I/O workers keep streaming the rest
  (true producer/consumer instead of the current ordered `pipe_wait` spin). This
  hides the per-expert latency behind the matmul of already-ready experts.
- **Pin a hot expert working set in the page cache / RAM** so `pread` becomes a
  pure memcpy. The model file is only 2.67 GB and fits in cache; the issue is
  the 12 GB dense + 6 GB reserve squeeze leaves it thrashing against the LRU. On
  the 96 GB HX 370 this is a non-issue and residency should approach 100%.
- **Reduce per-expert bytes:** the int4 experts are 18.9 MB each (3 tensors ×
  ~2 MB + scales). A coalesced single-shot read per expert (already done when
  the 3 tensors are file-contiguous; GLM's layout is not) would cut syscalls.

## Cross-box note (HX 370, 96 GB)
The user's *other* Strix Halo machine has 96 GB. The same architecture, so every
change here applies. Key differences that flip the optimum:
- Much larger RAM → the 2.5 GB `pc_b` reserve and dense 12 GB leave ~70 GB for
  expert residency. `cap` can be raised to dozens of experts/layer → hit rate
  should climb toward the lab's 100%, which **both** speeds decode *and* makes
  MTP drafting pay off (the 6x5090 / 96 GB HX 370 docs show MTP is negative
  only when experts miss). On that box `MTP_AUTO` should *stay on*.
- The `PC_GB` knob is the real lever there: lowering the reserve buys resident
  experts directly.
- The deeper producer/consumer change helps both boxes equally.

## Env reference (added)
| env | default | effect |
|---|---|---|
| `MTP_AUTO` | 1 | disable MTP drafting once expert-disk wait > 50% of decode |
| `PC_GB` | 2.5 | page-cache reserve in GB (floor 0.5); trade for expert LRU |

## Verification
- `test_attn_path2.c`: PASS (int4-float precision, maxrel ≤ 4e-3) — attention
  offload untouched, output coherent ("The capital of France is Paris.").
- Build: `cd c && make VULKAN=1 ARCH=native` clean.
- Numbers are single-run per config, greedy, warm cache; run-to-run variance not
  bounded (see METAL-M5MAX-PERF-REPORT.md determinism caveat — the same
  parallel-reduction non-determinism applies here under PIPE/OMP).
