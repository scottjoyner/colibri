#!/usr/bin/env bash
set -Eeuo pipefail
root=/home/joyner/provider-r9700/colibri-pool-fixture-20261017
out=/home/joyner/provider-r9700/colibri-pool-fixture-evidence-20261017
mkdir -p "$out"
cd "$root/c"
sha256sum tests/test_vk_tier > "$out/binary.sha256"
for pool in 0 1; do
    for port in 18320 18321; do curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$port/health"; done
    echo "START pool=$pool $(date -Is)" >> "$out/controller.log"
    /usr/bin/timeout 105s /usr/bin/env \
      COLI_VK_STAGED=1 COLI_VK_UP_IMPORT=1 COLI_VK_UP_POOL="$pool" \
      COLI_VK_TIER_FILL=0 OMP_NUM_THREADS=2 \
      ./tests/test_vk_tier shaders/qmatmul.spv > "$out/vulkan_pool_"$pool".log" 2>&1
    grep -q '^PASS$' "$out/vulkan_pool_"$pool".log"
    echo "PASS pool=$pool $(date -Is)" >> "$out/controller.log"
done
for port in 18320 18321; do curl -m 3 -fsS -o /dev/null "http://100.83.215.83:$port/health"; done
echo ALL_PASS >> "$out/controller.log"
