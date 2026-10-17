#!/usr/bin/env bash
set -Eeuo pipefail
d=/home/joyner/provider-r9700/colibri-fork-expert-trace-20261016/benchmarks/hardware-20261017
shard=/srv/joyner-gen5/models/glm53_colibri_i4/model-00002-of-00062.safetensors
test "$(findmnt -no UUID -T "$shard")" = b731381e-20b2-4c44-84a2-251b73970db1
for i in 1 2 3; do
  sample="$d/sample_$i"_128.txt
  test "$(wc -l < "$sample")" = 128
  for port in 18320 18321; do curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$port/health"; done
  if test "$i" = 2; then order="baseline imported"; else order="imported baseline"; fi
  for name in $order; do
    if test "$name" = imported; then bin="$d/bench_vulkan_imported_host"
    else bin="$d/bench_vulkan_expert_pipeline"; fi
    json="$d/imported_pair_$i"_"$name.json"
    test ! -e "$json"
    echo "RUN_START date=$(date -Is) trial=$i mode=$name"
    "$bin" "$shard" "$sample" 128 > "$json" 2> "$d/imported_pair_$i"_"$name.err"
    python3 -m json.tool "$json" >/dev/null
    echo "RUN_PASS date=$(date -Is) trial=$i mode=$name"
  done
done
for port in 18320 18321; do curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$port/health"; done
echo "ALL_PAIRED_RUNS_PASS $(date -Is)"
