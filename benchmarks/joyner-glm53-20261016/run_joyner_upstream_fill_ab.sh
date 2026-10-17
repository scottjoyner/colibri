#!/usr/bin/env bash
set -Eeuo pipefail
base=/home/joyner/provider-r9700/colibri-warmfill-smoke-20261017
d=/home/joyner/provider-r9700/colibri-upstream-fill-evidence-20261017
mkdir -p "$d"
exe="$base/c/glm53"
model=/srv/joyner-gen5/models/glm53_colibri_i4
stage=/srv/joyner-gen5/colibri-fork-dense-stage
test -x "$exe"
test "$(findmnt -no UUID -T "$stage")" = b731381e-20b2-4c44-84a2-251b73970db1
sha256sum "$exe" > "$d/binary.sha256"
for mode in history_only fill_opt_in; do
    if test "$mode" = history_only; then fill=0; else fill=1; fi
    for port in 18320 18321; do
        curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$port/health"
    done
    log="$d/glm53_$mode.log"
    test ! -e "$log"
    printf 'BEGIN %s %s\n' "$mode" "$(date -Is)" | tee -a "$d/controller.log"
    env GLM53_BOUNDED_DENSE=1 GLM53_DENSE_SPILL_DIR="$stage" \
      GLM53_BITS=4 GLM53_EXPERT_GB=0.05 GLM53_MAXT=64 COLI_VULKAN=1 COLI_VK_DEV=1 \
      COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=1 COLI_VK_TIER_GB=2 COLI_VK_TIER_WARM=1 \
      COLI_VK_TIER_FILL="$fill" COLI_VK_STAGED=1 COLI_VK_UP_IMPORT=0 \
      USAGE_SAVE=0 GLM53_VERBOSE=1 OMP_NUM_THREADS=2 \
      "$exe" --model "$model" --ids 11,21,13 --greedy 8 > "$log" 2>&1 &
    pid=$!
    completed=0
    previous_pressure=0
    for i in $(seq 1 65); do
        sleep 2
        if ! kill -0 "$pid" 2>/dev/null; then completed=1; break; fi
        available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        psi=$(awk '/^full / {split($2,v,"="); print v[2]}' /proc/pressure/memory)
        printf '%s mode=%s mem_available_kib=%s psi_full_avg10=%s\n' "$(date -Is)" "$mode" "$available" "$psi" >> "$d/pressure.log"
        high=$(awk -v p="$psi" 'BEGIN {print p>6.0 ? 1 : 0}')
        if test "$high" = 1 && test "$previous_pressure" = 1; then
            echo "SAFETY_STOP memory_pressure $mode" | tee -a "$d/controller.log"
            kill -TERM "$pid" || :
            wait "$pid" || :
            exit 13
        fi
        previous_pressure=$high
        if test "$available" -lt 230000; then
            echo "SAFETY_STOP low_available_memory $mode" | tee -a "$d/controller.log"
            kill -TERM "$pid" || :
            wait "$pid" || :
            exit 14
        fi
    done
    if test "$completed" = 0; then
        echo "SAFETY_STOP bounded_time $mode" | tee -a "$d/controller.log"
        kill -TERM "$pid" || :
        wait "$pid" || :
        exit 15
    fi
    wait "$pid"
    grep -qE '^decode 8 token in ' "$log"
    printf 'PASS %s %s\n' "$mode" "$(date -Is)" | tee -a "$d/controller.log"
done
for port in 18320 18321; do
    curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$port/health"
done
echo PAIR_COMPLETED | tee -a "$d/controller.log"
