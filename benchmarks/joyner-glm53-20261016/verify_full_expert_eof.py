#!/usr/bin/env python3
"""Read-only physical verification of safetensors EOF O_DIRECT+<4KiB tail handling."""
from __future__ import annotations
import hashlib,json,mmap,os,time
from pathlib import Path
import benchmark_complete_expert_qd2 as full
import benchmark_trace_predicted_reads as base

def main():
    base.safety()
    index=full.load_index()
    all_boundary=[]
    for key,records in index.items():
        for record in records:
            if record[-1]:
                all_boundary.append((key,record))
    if not all_boundary or len(all_boundary)>128:
        raise ValueError("NO_GO impossible EOF boundary count")
    fds={}
    plain={}
    matched=0
    total_tail=0
    direct_total=0
    checkpoint_before={}
    for key,(path,*_) in all_boundary:
        checkpoint_before[path]=(path.stat().st_size,path.stat().st_mtime_ns)
    try:
        for key,(path,offset,direct,tail_start,tail_len) in all_boundary:
            base.safety()
            if path not in fds:
                fds[path]=os.open(path,os.O_RDONLY|os.O_DIRECT|os.O_CLOEXEC)
                plain[path]=os.open(path,os.O_RDONLY|os.O_CLOEXEC)
            stream=plain[path]
            with path.open("rb") as f:
                header_len=int.from_bytes(f.read(8),"little")
                header=json.loads(f.read(header_len))
            # Find exactly the tensor from this expert whose last original
            # byte matches this descriptor's source shard EOF.
            matches=[]
            for name,t in header.items():
                if f".layers.{key[0]}.mlp.experts.{key[1]}." not in name:
                    continue
                first,last=t["data_offsets"]
                start=8+header_len+first
                if start+last-first==path.stat().st_size and start>=offset:
                    matches.append((start,last-first))
            if len(matches)!=1:
                raise ValueError("NO_GO ambiguous EOF tensor identity")
            start,size=matches[0]
            with mmap.mmap(-1,direct,access=mmap.ACCESS_WRITE) as buf:
                received=os.preadv(fds[path],[buf],offset)
                if received!=direct:raise OSError("NO_GO short boundary O_DIRECT")
                from_direct=buf[:]
            buffered_tail=os.pread(stream,tail_len,tail_start)
            if len(buffered_tail)!=tail_len:raise OSError("NO_GO short EOF tail")
            combined=from_direct+buffered_tail
            observed=combined[start-offset:start-offset+size]
            expected=os.pread(stream,size,start)
            if observed!=expected:
                raise RuntimeError("NO_GO direct/tail physical bytes mismatch")
            matched+=1
            direct_total+=received
            total_tail+=tail_len
        for path,(size,mtime) in checkpoint_before.items():
            stat=path.stat()
            if stat.st_size!=size or stat.st_mtime_ns!=mtime:
                raise RuntimeError("NO_GO model source mutation detected")
    finally:
        for fd in fds.values():os.close(fd)
        for fd in plain.values():os.close(fd)
    print(json.dumps({
        "schema":1,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "pin":base.PIN,"complete_expert_index":len(index),
        "EOF_tensor_windows_tested":len(all_boundary),
        "byte_for_byte_matches":matched,"model_mutations_detected":0,
        "O_DIRECT_bytes":direct_total,"strictly_read_only_buffered_EOF_tail_bytes":total_tail,
        "tail_bytes_max":max(x[1][-1] for x in all_boundary),
        "source_shards_touched":len(checkpoint_before),
        "no_native_router":True,"production_untouched":True
    },indent=2))
if __name__=="__main__":
    try:main()
    except Exception as exc:
        import sys
        print("NO_GO:",type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(2)
