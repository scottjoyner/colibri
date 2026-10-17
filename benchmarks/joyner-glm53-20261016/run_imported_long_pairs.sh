#!/usr/bin/env bash
set -Eeuo pipefail
d=/home/joyner/provider-r9700/colibri-fork-expert-trace-20261016/benchmarks/hardware-20261017
shard=/srv/joyner-gen5/models/glm53_colibri_i4/model-00002-of-00062.safetensors
sample="$d/sample_384_unique.txt"
test "$(wc -l < "$sample")" = 384
test "$(sort -u "$sample" | wc -l)" = 384
for trial in 1 2; do
    if test "$trial" = 1; then order="baseline imported"; else order="imported baseline"; fi
    for mode in $order; do
        for port in 18320 18321; do curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$port/health"; done
        if test "$mode" = baseline; then bin="$d/bench_vulkan_expert_pipeline"
        else bin="$d/bench_vulkan_imported_host"; fi
        result="$d/imported_long_$trial"_"$mode.json"
        test ! -e "$result"
        echo "BEGIN trial=$trial mode=$mode $(date -Is)"
        "$bin" "$shard" "$sample" 384 > "$result" 2> "$d/imported_long_$trial"_"$mode.err"
        python3 -m json.tool "$result" >/dev/null
        echo "PASS trial=$trial mode=$mode $(date -Is)"
    done
done
echo "ALL_384_TRANSFER_PAIRS_PASS"
