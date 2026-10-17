#!/usr/bin/env python3
"""Read-only same-offset O_DIRECT expert QD sweep on Joyner Gen5 NVMe.
Safety: hash-pinned GLM checkpoint, 4KiB-aligned, no writes to model, bounded run.
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import bench_expert_direct_io as direct

ORDERS=((1,2,4,8),(8,4,2,1),(2,1,8,4))

def block_read_bytes():
    # Disk-wide delta is a *rough* witness; concurrent processes can read the SSD.
    data=Path("/sys/block/nvme0n1/stat").read_text().split()
    return int(data[2])*512

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-round",type=int,default=256)
    p.add_argument("--rounds",type=int,default=3)
    args=p.parse_args()
    if not 32<=args.per_round<=256 or not 1<=args.rounds<=3:
        p.error("per-round 32..256, rounds 1..3")
    root=direct.ROOT
    if os.popen("findmnt -n -T /srv/joyner-gen5 -o SOURCE").read().strip()!="/dev/nvme0n1":
        raise SystemExit("NO_GO unexpected Gen5 disk")
    provenance=json.loads((root/".source-revision.json").read_text())
    assert provenance=={"repo":"zai-org/GLM-5.3-Flash","revision":"46a050bec13002ad29ee795c16e61cf4fa3c4a9c"}
    manifest=json.loads((root/".converted.json").read_text())
    checksums=json.loads((root/".converted-sha256.json").read_text())
    if set(manifest)!=set(checksums) or len(manifest)!=62:
        raise SystemExit("NO_GO manifest mismatch")
    candidates=[]
    for shard in sorted(manifest):
        if not (shard.startswith("model-") and shard.endswith(".safetensors")):
            raise SystemExit("NO_GO invalid model shard")
        candidates.extend(direct.examine(root/shard))
    if len(candidates)<args.per_round*args.rounds:
        raise SystemExit("NO_GO too few expert regions")
    # Use identical *unrepeated* regions in each group's QD configurations.
    rng=random.Random(0xC011B120261017)
    selected=rng.sample(candidates,args.per_round*args.rounds)
    shas=[hashlib.sha256(("\n".join(f"{item[0]}|{item[2]}|{item[3]}" for item in selected[r*args.per_round:(r+1)*args.per_round])).encode()).hexdigest() for r in range(args.rounds)]
    fds={}
    measures=[]
    try:
        for path in sorted({x[0] for x in selected}):
            fds[path]=os.open(path,os.O_RDONLY|os.O_DIRECT|os.O_CLOEXEC)
        for round_id in range(args.rounds):
            samples=selected[round_id*args.per_round:(round_id+1)*args.per_round]
            for qd in ORDERS[round_id]:
                before=block_read_bytes()
                data=direct.bench(samples,fds,qd)
                after=block_read_bytes()
                data.update(round=round_id,requested_workers=qd,physical_read_delta_GB=round((after-before)/1e9,3),
                            aligned_region_set_sha256=shas[round_id])
                measures.append(data)
                print(f"round={round_id} workers={qd} GBps={data['direct_GBps']:.3f} p99_ms={data['p99_ms']:.3f}",flush=True)
    finally:
        for fd in fds.values():os.close(fd)
    grouped={}
    for qd in sorted({x["workers"] for x in measures}):
        xs=[r for r in measures if r["workers"]==qd]
        grouped[str(qd)]={"median_GBps":round(statistics.median(r["direct_GBps"] for r in xs),3),
                          "median_p99_ms":round(statistics.median(r["p99_ms"] for r in xs),3),
                          "rounds":len(xs)}
    report={"schema":1,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "device":"/dev/nvme0n1","data":"SHA-manifest-listed GLM-5.3 real int4 expert tensors",
            "pinned_revision":provenance["revision"],"manifest_shards":len(manifest),
            "candidate_expert_tensor_regions":len(candidates),"per_round":args.per_round,
            "rounds":args.rounds,"total_O_DIRECT_GB":round(sum(x["read_GB"] for x in measures),3),
            "same_offsets_within_round":True,"no_repeated_offsets_across_rounds":True,
            "measurement_notes":["whole block device physical sector counter includes concurrent processes",
                                 "O_DIRECT bypasses Linux page cache, not NVMe controller cache",
                                 "throughput is storage-only, no decode/GPU contention",
                                 "individual p99 from <=256 samples per point; tails require more validation"],
            "runs":measures,"medians":grouped}
    out=Path("/home/joyner/provider-r9700/colibri-pool-fixture-evidence-20261017/gen5_paired_expert_qd_20261017.json")
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("OUTPUT",out,flush=True)

if __name__=="__main__":main()
