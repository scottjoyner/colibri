#!/usr/bin/env bash
set -Eeuo pipefail
d=/home/joyner/provider-r9700/colibri-fork-expert-trace-20261016/benchmarks/hardware-20261017
runner=/home/joyner/provider-r9700/colibri-fork-expert-trace-20261016/c/glm53
model=/srv/joyner-gen5/models/glm53_colibri_i4
stage=/srv/joyner-gen5/colibri-fork-dense-stage
test "$(findmnt -no UUID -T "$stage")" = b731381e-20b2-4c44-84a2-251b73970db1
test -x "$runner"
for mode in staged_no_import staged_host_import; do
  if test "$mode" = staged_no_import; then imported=0; else imported=1; fi
  for p in 18320 18321; do curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$p/health"; done
  test ! -e "$d/full_glm_$mode.log"
  echo "$(date -Is) BEGIN $mode"
  /usr/bin/timeout 205s /usr/bin/env GLM53_BOUNDED_DENSE=1 GLM53_DENSE_SPILL_DIR="$stage"     GLM53_BITS=4 GLM53_EXPERT_GB=0.05 GLM53_MAXT=64 COLI_VULKAN=1 COLI_VK_DEV=1     COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=1 COLI_VK_TIER_GB=2 COLI_VK_TIER_WARM=0     COLI_VK_STAGED=1 COLI_VK_UP_IMPORT="$imported" USAGE_SAVE=0 GLM53_VERBOSE=1     OMP_NUM_THREADS=2 "$runner" --model "$model" --ids 11,21,13 --greedy 8     > "$d/full_glm_$mode.log" 2>&1
  echo "$(date -Is) PASS $mode"
done
for p in 18320 18321; do curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$p/health"; done
echo "$(date -Is) PAIRED_GLM_IMPORT_TEST_FINISHED"
