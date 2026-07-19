#!/usr/bin/env bash
# colibri optimization-round benchmark harness for the HX 370 (Strix Point, 96 GB).
#
# Discipline (per docs/experiments/glm52-hx370-2026-07-18.md + 6x5090 lab):
#   - fixed greedy prompt ("water cycle") so MTP/sampling comparisons are valid
#   - DRAFT=0 (MTP is net-negative on this disk-bound box, proven)
#   - core pinning: OMP_NUM_THREADS=12 OMP_PROC_BIND=spread OMP_PLACES=cores
#   - CPU-only (COLI_GPUS=none); the 890M iGPU is not targeted by colibri
#
# Usage:
#   ./bench_hx370.sh            # run the default sweep and append to results log
#   RAM_GB=60 TOPP=0.7 NGEN=96 ./bench_hx370.sh
#
set -u
cd "$(dirname "$0")"

MODEL=${SNAP:-${COLI_MODEL:-/media/scott/SSD_4TB/models-fast/GLM-5.2-colibri-int4-with-int8-mtp}}
GLM=${GLM:-./glm}

RAM_GB=${RAM_GB:-50}
TOPP=${TOPP:-0}
TOPK=${TOPK:-1}
N=${N:-96}                 # tokens to generate
MTP=${MTP:-0}              # forced off: MTP is net-negative on this disk-bound box
PROMPT=${PROMPT:-"The water cycle is the continuous movement of water on, above, and below the surface of the Earth. Explain how it works."}

# core pinning per winning 6x5090 config (12 physical cores; HX370 has 12 cores)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
# CPU-only binary: do NOT set COLI_GPUS (the engine treats any value as a CUDA request).
unset COLI_GPUS 2>/dev/null || true

RUN_ID=$(date +%Y%m%d-%H%M%S)
RESULTS=${RESULTS:-bench_hx370_results.tsv}
[ -f "$RESULTS" ] || printf 'run_id\tram_gb\ttopp\tn\tcores\tdecode_tok_s\thit_rate\tdisk_wait_s\tdecode_s\trss_gb\n' > "$RESULTS"

echo "=== bench $RUN_ID | ram=$RAM_GB topp=$TOPP n=$N cores=$OMP_NUM_THREADS ==="
echo "    prompt: ${PROMPT:0:48}..."

# SNAP=<model> ./glm <ctx> <batch> <ngen>  -> here batch=1 (decode), ctx small
# Capture stderr (stats) + stdout (text). Use --topp via env (TOPP) if supported,
# else fall back to args. colibri reads TOPP/TEMP from env.
export RAM_GB
export TOPP
export NGEN=$N
export DRAFT=0
export MTP=$MTP
export TEMP=0
export TOPK=1
export PROMPT

OUT=$(SNAP="$MODEL" timeout 600 "$GLM" 512 1 "$N" 2>&1)
RC=$?

if [ $RC -ne 0 ]; then
  echo "RUN FAILED (rc=$RC)"; echo "$OUT" | tail -20; exit 1
fi

# Parse the summary line colibri prints:
#   "prefill N tokens in Xs | decode N tokens in Xs (X tok/s) | expert hit rate X% | RSS X GB"
SUMMARY=$(echo "$OUT" | grep -E 'decode .* tokens in' | tail -1)
DECODE=$(echo "$SUMMARY" | grep -oE 'decode [0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s\)' | grep -oE '[0-9.]+ tok/s' | grep -oE '[0-9.]+')
HIT=$(echo "$SUMMARY" | grep -oE 'expert hit rate [0-9.]+%' | grep -oE '[0-9.]+')
DECODE_S=$(echo "$SUMMARY" | grep -oE 'decode [0-9]+ tokens in [0-9.]+s' | grep -oE '[0-9.]+s' | grep -oE '[0-9.]+')
RSS=$(echo "$SUMMARY" | grep -oE 'RSS [0-9.]+ GB' | grep -oE '[0-9.]+')
DISK_WAIT=$(echo "$OUT" | grep -E 'PROFILE: expert-disk' | tail -1 | grep -oE 'expert-disk [0-9.]+s service / [0-9.]+s wait' | grep -oE 'service / [0-9.]+s wait' | grep -oE '[0-9.]+s' | tail -1 | grep -oE '[0-9.]+')

DECODE=${DECODE:-NA}; HIT=${HIT:-NA}; DISK_WAIT=${DISK_WAIT:-NA}; DECODE_S=${DECODE_S:-NA}; RSS=${RSS:-NA}

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$RUN_ID" "$RAM_GB" "$TOPP" "$N" "$OMP_NUM_THREADS" "$DECODE" "$HIT" "$DISK_WAIT" "$DECODE_S" "$RSS" >> "$RESULTS"

echo "    -> decode=${DECODE} tok/s  hit=${HIT}%  disk_wait=${DISK_WAIT}s  decode_s=${DECODE_S}s  rss=${RSS}GB"
echo "    saved to $RESULTS"
