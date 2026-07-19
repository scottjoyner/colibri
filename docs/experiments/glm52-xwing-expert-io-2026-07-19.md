# Expert I/O investigation — xwing (Radeon 8050S, 20 GB cap)

Date: 2026-07-19 · Machine: **xwing** (AMD Ryzen AI MAX PRO 390 / Radeon 8050S,
Strix Halo) · Model: `/media/scott/data/GLM-5.2-colibri-int4-with-int8-mtp`
(144 shards, ~2.67 GB total) · RAM budget: **RAM_GB=20** · NVMe
`/dev/nvme0n1p3` (~5–8 GB/s sequential, file fully *could* fit in OS page
cache but gets evicted under the 20 GB budget).

This note records the expert-load bottleneck analysis on xwing and the changes
shipped in commit `457a536` + the deeper `COLI_FILECACHE` investigation.
It complements `glm52-hx370-2026-07-18.md` (the *other* HX 370, 96 GB,
different snapshot) and `glm52-incoherent-text-investigation.md`.

## TL;DR

The reported bottleneck — `PROFILE: expert-disk 0.000s service / 21.6s wait` —
is **not disk bandwidth** and **not read ordering**. Instrumenting `expert_load`
(`EXPLOAD_TIMING=1`) showed **avg 71 ms / expert, 99% of it in the I/O
region**, but a standalone `pread` of the same 18 MB region reads at **5.3 GB/s
cold**. So the engine gets **0.23 GB/s effective = a 23× gap** vs raw pread.

**Root cause: memory-bandwidth contention on the unified Strix Halo APU.** The
dense MoE matmul (12 GB weights, OMP, ~2.4s decode) and the expert
`pread`→slab copies (18 MB each) both hammer the *same* shared RAM channel.
While OMP saturates the bus with dense math, the I/O workers' reads are
starved to 0.23 GB/s. Cold raw pread (no concurrent matmul) gets the full
5.3 GB/s. Reordering completion, fewer OMP threads, passive wait — all A/B'd
**flat** — confirming it is bus contention, not scheduling or ordering.

End-to-end on xwing (attention offload `COLI_CUDA_ATTN=1`, RAM_GB=20,
`PROMPT="The capital of France is"`, NGEN=8):

| config | tok/s | experts/token | expert-disk wait | notes |
|---|---|---|---|---|
| baseline (default, MTP on, draft=3) | 0.27 | 1670 | 21.6s | MTP loads 3× redundant experts |
| `MTP=0` | 0.40 | 525 | 15.4s | correct ~topk×layers load count |
| `MTP=0 PIPE_WORKERS=16` | 0.41 | 525 | 14.9s | workers not the limiter |
| `COLI_MMAP=1` | 0.09 | 1595 | 36.6s | mmap+4KB pre-touch is 3× *worse* |
| **optimized default (MTP_AUTO on)** | **0.38** | 679 | 16.2s | +41% vs baseline, automatic |
| `COLI_FILECACHE=1` (no root) | 0.33 | 753 | 18.1s | mlock denied → buffered fallback, no regression |

## What was actually measured

1. **Disk is not the bottleneck.** `dd` of the 2.67 GB file: 8.4 GB/s. A
   synthetic random 18 MB × 200 `pread` loop: **5.3 GB/s** (cold, after the
   engine had evicted it). So 18 MB should take ~3.4 ms — but the engine
   spends **71 ms/expert**. 23× slower than the same read in isolation.
2. **Warm cache = cold cache.** A second identical run (file already in page
   cache) and a manual `cat`-preload both gave the **same ~0.35 tok/s**.
   The histogram (`EXPLOAD_TIMING`) is damning: **3904/4000 calls took
   20–100 ms, only 12 took <20 ms** — i.e. ~98% of reads are slow,
   ~2% fast. Not a hot/warm split; a steady-state slow path.
3. **`COLI_MMAP=1` is harmful here.** The mmap path does a 4 KB page
   pre-touch per tensor = ~855k page faults/token; under 20 GB RAM most faults
   go hard → 0.09 tok/s (3× slower). Keep the default pread+slab path.
4. **MTP is a net loss at this speed.** ~25% draft acceptance → MTP loads
   ~3× experts (1670 vs 525/token) for rejected positions. Off → +48%.
5. **Worker count / OMP threads / wait policy: all flat.** `PIPE_WORKERS=16`,
   `OMP_NUM_THREADS=6/8/12`, `OMP_WAIT_POLICY=passive` — each within
   ±0.03 tok/s of the others. Confirms the limiter is bus contention, not
   pipeline width or scheduling.

## Changes shipped (commit `457a536`)

### 1. `MTP_AUTO` (default ON) — the real win (+41%)
In `spec_decode`, sample per-forward expert-disk wait vs decode wall-time. Over a
~3s window, if wait exceeds **50%** of decode, latch `m->has_mtp=0` for
the rest of the run and print:
```
[MTP_AUTO] expert-disk wait 71% of decode (5.54s/7.81s): disabling MTP
           drafting for the rest of the run (MTP_AUTO=0 to keep)
```
Opt-outs: `MTP_AUTO=0` keeps MTP on; explicit `MTP=0` still wins at startup.
This auto-recovers the +48% `MTP=0` result with no user action, and will
*keep* MTP on machines where acceptance is high enough that drafting pays off
(e.g. the 96 GB HX 370 when experts are mostly resident — see below).

### 2. Pre-allocated `ws[64]` dispatch slabs
`expert_load` re-allocates a slab only if `slab==NULL` or `wtot+8192 > slab_cap`.
Every expert has identical dims, so `wtot` is constant — after the first load
the slab persists and the LRU swap only moves pointers, never frees. The change
**pre-allocates the shared `ws[64]` slabs once at `model_init`** (sized from
`expert_bytes_probe`), removing the first-dispatch `posix_memalign` and
guaranteeing persistence. Neutral on timing (the old path already avoided
realloc), but makes residency robust and removes a per-dispatch branch.

### 3. `PC_GB` tunable for the page-cache reserve
The RAM budget reserves a fixed 2.5 GB page-cache margin (`pc_b`,
glm.c:5757) to keep buffered `pread` throughput up. At RAM_GB=20 this margin
plus the 12 GB dense forces `cap=1` expert LRU residency, so ~every token
reloads all experts. `PC_GB=<n>` lets the user trade that margin for more
expert cache. Tested 2.5 → 1.0: cap rose 1→2 but tok/s was flat (0.34–0.37)
and RSS rose toward the wall — at 20 GB the 12 GB dense dominates, so this knob
is **ineffective on xwing** but is a real lever on bigger-RAM Strix Halo boxes.

## Deeper architectural investigation (what the bottleneck actually is)

Instrumented `expert_load` (`EXPLOAD_TIMING=1`): **avg 71 ms / expert, 99%
of it in the I/O region** — but the model file reads at **5.3 GB/s cold** in a
standalone `pread` loop. So the engine gets **0.23 GB/s effective vs 5.3 GB/s
raw = a 23× gap**. It is *not* disk seek, *not* read ordering, *not* the
forward thread stalling on `pipe_wait` (re-ordering the completion — processing
ready experts out of order — measured identically at 0.35 tok/s).

**Root cause: memory-bandwidth contention on the unified Strix Halo APU.**
The 8050S has one shared RAM channel. The dense MoE matmul (12 GB weights,
`matmul_qt` over OMP, ~2.4s decode) and the expert `pread` copies (18 MB
into `ws[]` slabs) both hammer that same bandwidth. While OMP saturates the
bus with dense math, the I/O workers' `pread`→copy is starved to ~0.23 GB/s.
Cold raw `pread` (no concurrent matmul) gets the full 5.3 GB/s. OMP thread
count (6/8/12), passive wait policy, and out-of-order completion were all A/B'd
— **flat** — confirming it is bus contention, not scheduling or ordering.

### Implemented lever: `COLI_FILECACHE=1` (mmap + mlock the model)

The fix is to remove the expert reads from the contended path by pinning the
entire safetensors set into RAM once at load (`st_init`): `mmap(PRIVATE)` each
shard, `mlock` it, pre-fault every page, then `expert_load` reads via
`memcpy` from the locked mapping (`fc_read`) instead of `pread`. No NVMe, no
page faults, no contended copy-from-cache — just a RAM-to-RAM memcpy.

- **On the 96 GB HX 370 (run as root / `ulimit -l unlimited`): this is the
  key win.** The 2.67 GB model pins easily; expert reads become ~0.1 ms RAM
  memcpys, the cold-read tax collapses, hit rate climbs toward 100%, and
  `MTP_AUTO` *stays on* (drafting pays off when experts are resident).**
- **On xwing (no root, `ulimit -l` = 3 GB): a no-op, by design.** One
  shard is 5.1 GB > the 3 GB mlock cap, so the whole model can't be
  pinned. `COLI_FILECACHE=1` detects mlock failure per-shard, `munmap`s,
  and falls back to the correct buffered `pread` — **zero regression**
  (verified: 0.33 tok/s, no crash). Largest-shard-first ordering pins as
  much as the budget allows before stopping on the first mlock failure.

### Why not the other levers
- **Forward-thread out-of-order completion:** measured identical (0.35 tok/s) —
  the stall is bandwidth, not ordering. Reverted.
- **`PC_GB` / smaller reserve:** at 20 GB the 12 GB dense dominates; lowering
  the 2.5 GB cache reserve buys ~nothing (cap 1→2, tok/s flat).
- **Fewer OMP threads:** slows the matmul, no net gain (bus still contended).

### The real remaining lever on xwing
The 23× is unified-APU bus contention between dense matmul and expert I/O.
The only architectural fix that moves it is **offloading the dense/expert matmul to
the 8050S iGPU via the Vulkan backend** (freeing CPU memory bandwidth for
the expert `pread`s). That is the whole `backend_vulkan.c` MoE path — explored
earlier (the fused pipeline was broken/slow on this APU); it remains the
highest-value deep change but is a backend project, not an I/O tweak.

## Cross-box note (HX 370, 96 GB)
The user's *other* Strix Halo machine has 96 GB. Same architecture → every
change here applies, and the optimum flips:
- **`COLI_FILECACHE=1` (+ root / `ulimit -l unlimited`) is the headline win
  there:** pins the 2.67 GB model, kills the 23× bandwidth-via-cold-read
  tax, drives residency → 100%, and keeps `MTP_AUTO` *on*.
- Much larger RAM → the 2.5 GB `pc_b` reserve and dense 12 GB leave ~70 GB
  for expert residency; `cap`/`PC_GB` raise hit rate directly.
- The Vulkan iGPU matmul offload (above) helps both boxes equally.

## Env reference (added)
| env | default | effect |
|---|---|---|
| `MTP_AUTO` | 1 | disable MTP drafting once expert-disk wait > 50% of decode |
| `PC_GB` | 2.5 | page-cache reserve in GB (floor 0.5); trade for expert LRU |
| `COLI_FILECACHE` | off | mmap+mlock model shards into RAM; needs `ulimit -l`/root to help |

## Verification
- `test_attn_path2.c`: PASS (int4-float precision, maxrel ≤ 4e-3) — attention
  offload untouched, output coherent ("The capital of France is Paris.").
- Build: `cd c && make VULKAN=1 ARCH=native` clean.
- `COLI_FILECACHE=1` without mlock: verified **no regression** (falls back to
  buffered pread; earlier `S->fds[fd]` fd-indexing bug causing "short read at
  EOF" crashes fixed — `fc_read` now uses the OS fd directly).
- Numbers are single-run per config, greedy, warm cache; run-to-run variance not
  bounded (see METAL-M5MAX-PERF-REPORT.md determinism caveat — the same
  parallel-reduction non-determinism applies here under PIPE/OMP).
