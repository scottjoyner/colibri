#!/usr/bin/env python3
"""Read-only O_DIRECT sampler for REAL converted Colibri GLM expert tensors.
Does not open confidential archives, perform GPU dispatch or modify model shards.
"""
import argparse
import concurrent.futures
import json
import math
import mmap
import os
import random
import statistics
import struct
import threading
import time
from pathlib import Path

ROOT=Path("/srv/joyner-gen5/models/glm53_colibri_i4")
ALIGN=4096
TLS=threading.local()

def roundup(v):return ((v+ALIGN-1)//ALIGN)*ALIGN
def examine(path):
    with path.open("rb") as stream:
        b=stream.read(8)
        if len(b)!=8: raise ValueError("truncated file")
        length=struct.unpack("<Q",b)[0]
        if length>50_000_000: raise ValueError("header too large")
        h=json.loads(stream.read(length))
    result=[]
    data_offset=8+length
    file_size=path.stat().st_size
    for key,item in h.items():
        if key=="__metadata__" or ".mlp.experts." not in key or key.endswith(".qs") or item.get("dtype")!="U8":continue
        start,end=item["data_offsets"]
        absolute_start=data_offset+start
        absolute_end=data_offset+end
        if absolute_end>file_size or absolute_start<0 or absolute_end<=absolute_start:
            raise ValueError("invalid expert offsets")
        aligned_start=(absolute_start//ALIGN)*ALIGN
        aligned_end=roundup(absolute_end)
        if aligned_end>file_size:continue
        result.append((str(path),key,aligned_start,aligned_end-aligned_start,absolute_end-absolute_start))
    return result

def one(item, fds, max_bytes):
    path,name,offset,count,useful=item
    if not hasattr(TLS,"buf"): TLS.buf=mmap.mmap(-1,max_bytes,access=mmap.ACCESS_WRITE)
    t0=time.perf_counter()
    size=os.preadv(fds[path],[memoryview(TLS.buf)[:count]],offset)
    elapsed=time.perf_counter()-t0
    if size!=count: raise IOError(f"short read {size} of {count}")
    return elapsed,count,useful

def bench(samples,fds,workers):
    max_bytes=max(x[3] for x in samples)
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(lambda item:one(item,fds,max_bytes),samples))
    wall=time.perf_counter()-start
    latencies=sorted(x[0]*1000 for x in results)
    def pct(p):
        return round(latencies[min(len(latencies)-1,max(0,math.ceil(len(latencies)*p)-1))],3)
    io_bytes=sum(x[1] for x in results)
    useful_bytes=sum(x[2] for x in results)
    return {"workers":workers,"completed":len(samples),"read_GB":round(io_bytes/1e9,3),
            "payload_GB":round(useful_bytes/1e9,3),"elapsed_s":round(wall,3),
            "direct_GBps":round(io_bytes/wall/1e9,3),
            "p50_ms":pct(.5),"p95_ms":pct(.95),"p99_ms":pct(.99)}

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--requests",type=int,default=128)
    args.add_argument("--iterations",type=int,default=1)
    args.add_argument("--workers",default="1,4",help="comma-separated direct-I/O reader counts, range 1..16")
    options=args.parse_args()
    if not (8<=options.requests<=256): args.error("requests must be 8..256")
    if not (1<=options.iterations<=16): args.error("iterations must be 1..16")
    worker_counts=[int(x) for x in options.workers.split(",")]
    if not worker_counts or len(worker_counts)!=len(set(worker_counts)) or any(n<1 or n>16 for n in worker_counts):args.error("workers must be unique numbers from 1 to 16")
    device=os.popen("findmnt -n -T /srv/joyner-gen5 -o SOURCE").read().strip()
    if device!="/dev/nvme0n1": raise SystemExit("NO-GO: unexpected Gen5 mount")
    provenance=json.loads((ROOT/".source-revision.json").read_text())
    if provenance!={"repo":"zai-org/GLM-5.3-Flash","revision":"46a050bec13002ad29ee795c16e61cf4fa3c4a9c"}:
        raise SystemExit("NO-GO: wrong source revision")
    done=json.loads((ROOT/".converted.json").read_text())
    hashes=json.loads((ROOT/".converted-sha256.json").read_text())
    if len(done)<1 or set(done)!=set(hashes):raise SystemExit("NO-GO: manifests disagree")
    entries=[]
    for name in done:
        if not name.startswith("model-") or not name.endswith(".safetensors"):
            raise SystemExit("NO-GO: unexpected shard filename")
        entries.extend(examine(ROOT/name))
    if len(entries)<options.requests:
        raise SystemExit("NO-GO: too few real expert tensors")
    fds={}
    try:
        for path in {item[0] for item in entries}:
            fds[path]=os.open(path,os.O_RDONLY|os.O_DIRECT)
        count=options.requests*options.iterations
        selected={workers:random.Random(1253+workers).choices(entries,k=count) for workers in worker_counts}
        payload=[x[4] for x in entries]
        out={"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
             "source":"verified manifest-listed GLM-5.3-Flash Colibri int4 expert tensors",
             "source_shards":len(done),"sample_candidates":len(entries),
             "median_tensor_MiB":round(statistics.median(payload)/1048576,3),
             "min_tensor_MiB":round(min(payload)/1048576,3),
             "max_tensor_MiB":round(max(payload)/1048576,3),
             "tests":[bench(selected[workers],fds,workers) for workers in worker_counts]}
        print(json.dumps(out,indent=2))
    finally:
        for fd in fds.values():os.close(fd)

if __name__=="__main__":main()
