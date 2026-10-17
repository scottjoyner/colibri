#!/usr/bin/env bash
set -Eeuo pipefail
d=/home/joyner/provider-r9700/colibri-fork-expert-trace-20261016/benchmarks/hardware-20261017
source_file=/srv/joyner-gen5/models/glm53_colibri_i4/model-00002-of-00062.safetensors
test "$(findmnt -no UUID -T "$source_file")" = b731381e-20b2-4c44-84a2-251b73970db1
test -f /srv/joyner-gen5/models/glm53_colibri_i4/.converted-sha256.json
python3 - "$d" <<'PY'
from pathlib import Path
import random,sys
sys.path.insert(0,"/home/joyner/provider-r9700/colibri-glm53-eval")
from bench_expert_direct_io import examine
d=Path(sys.argv[1])
f=Path("/srv/joyner-gen5/models/glm53_colibri_i4/model-00002-of-00062.safetensors")
rows=examine(f)
eligible=sorted(set(x[2] for x in rows if x[3]>=4194304))
subset=random.Random(202610170121).sample(eligible,384)
for k in range(3):
    (d/f"sample_{k+1}_128.txt").write_text("".join(str(x)+"\n" for x in subset[k*128:(k+1)*128]))
print("UNIQUE_EXPERT_SAMPLES",len(set(subset)),flush=True)
PY
for run in 1 2 3; do
  for p in 18320 18321; do
    curl -m 3 -sf -o /dev/null "http://100.83.215.83:$p/health"
  done
  if [ "$run" = 2 ]; then order="serial pipeline"; else order="pipeline serial"; fi
  for mode in $order; do
    echo "RUN_BEGIN $run $mode $(date -Is)"
    binary="$d/bench_vulkan_expert_transfer"
    test "$mode" != pipeline || binary="$d/bench_vulkan_expert_pipeline"
    "$binary" "$source_file" "$d/sample_$run""_128.txt" 128 \
      > "$d/paired_$run""_$mode.json" 2> "$d/paired_$run""_$mode.err"
    python3 -m json.tool "$d/paired_$run""_$mode.json" >/dev/null
    echo "RUN_PASS $run $mode $(cat "$d/paired_$run""_$mode.json")"
  done
done
for p in 18320 18321; do
  curl -m 3 -sf -o /dev/null "http://100.83.215.83:$p/health"
done
echo "ALL_PAIRED_RUNS_FINISHED $(date -Is)"

