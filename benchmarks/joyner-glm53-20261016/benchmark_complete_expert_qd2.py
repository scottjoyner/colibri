#!/usr/bin/env python3
"""Physical full-six-tensor GLM expert shadow reads selected by frozen routing.

Synchronous per-event, QD2 across experts, O_DIRECT 4KiB aligned.
No GPU memory, production router, model weight, or native logits are modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import benchmark_trace_predicted_reads as base
from calibrated_prefetch_budget import fit
from bounded_shadow_reader import BoundedShadowReader
from replay_expert_prefetch import read_trace,compile_stats
from shadow_tier_plan import Limits,plan

FOUR=4*1024*1024
HALF=512*1024
LOGICAL_EXPERT_BYTES=3*FOUR+3*HALF
TENSORS=(
    ("down_proj.weight",FOUR),
    ("down_proj.weight.qs",HALF),
    ("gate_proj.weight",FOUR),
    ("gate_proj.weight.qs",HALF),
    ("up_proj.weight",FOUR),
    ("up_proj.weight.qs",HALF))
WINDOW=4096
STRICT_TENSOR=set(TENSORS)

def safe_aligned_span(start:int,length:int,file_bytes:int)->tuple[int,int,int,int]:
    """Physical O_DIRECT range plus at most one small read-only EOF tail."""
    if start<0 or length<=0 or start+length>file_bytes:
        raise ValueError("NO_GO expert tensor range exceeds source file")
    aligned=(start//WINDOW)*WINDOW
    offset_in_page=start-aligned
    count=((offset_in_page+length+WINDOW-1)//WINDOW)*WINDOW
    direct=min(count,((file_bytes-aligned)//WINDOW)*WINDOW)
    remaining=max(0,start+length-(aligned+direct))
    if direct<WINDOW or remaining>=WINDOW or aligned+direct>file_bytes:
        raise ValueError("NO_GO expert alignment or EOF tail invalid")
    return aligned,direct,aligned+direct,remaining

def load_index():
    # This independent check also validates exact source pin, manifest and NVMe mount.
    assert len(base.validate_model())==12384
    root=base.MODEL
    done=json.loads((root/".converted.json").read_text())
    experts={}
    for shard in sorted(done):
        file=root/shard
        with file.open("rb") as handle:
            raw=handle.read(8)
            if len(raw)!=8:raise ValueError("bad safetensor header prefix")
            n=struct.unpack("<Q",raw)[0]
            if n>50_000_000:raise ValueError("unreasonable safetensor header")
            header=json.loads(handle.read(n))
        file_bytes=file.stat().st_size
        for name,tensor in header.items():
            m=base.PATTERN.match(name)
            if m is None:
                # The name suffix changes but namespace stays constant.
                import re
                m=re.match(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)$",name)
            if not m:continue
            layer,eid=int(m.group(1)),int(m.group(2))
            suffix=name.split(".mlp.experts.",1)[1].split(".",1)[1]
            sizes=dict(TENSORS)
            if suffix not in sizes:continue
            expected_dtype="F32" if suffix.endswith(".qs") else "U8"
            if tensor.get("dtype")!=expected_dtype:
                raise ValueError("unexpected full expert dtype")
            first,last=tensor["data_offsets"]
            if last-first!=sizes[suffix]:raise ValueError("unexpected expert tensor size")
            start=8+n+first
            aligned,direct,tail_start,remaining=safe_aligned_span(
                start,last-first,file_bytes)
            bucket=experts.setdefault((layer,eid),{})
            if suffix in bucket:raise ValueError("duplicate full expert tensor")
            bucket[suffix]=(file,aligned,direct,tail_start,remaining)
    if len(experts)!=12384:raise ValueError(f"incomplete expert index {len(experts)}")
    for key,tensors in experts.items():
        if set(tensors)!=set(dict(TENSORS)):
            raise ValueError(f"incomplete six-tensor expert {key}")
    return {key:tuple(tensors[suffix] for suffix,_ in TENSORS)
            for key,tensors in experts.items()}

def disk_sector_bytes():
    return int(Path("/sys/block/nvme0n1/stat").read_text().split()[2])*512

def physically_read_expert(ignored_fd,expert_key, index,fds,tail_fds):
    record=index[expert_key]
    ms=[]
    bytes_read=0
    tail_bytes=0
    tails=0
    begun=time.perf_counter()
    for path,offset,count,tail_start,tail_count in record:
        if offset%WINDOW or count%WINDOW or count<WINDOW or tail_count>=WINDOW:
            raise ValueError("invalid O_DIRECT expert IO")
        buf=mmap.mmap(-1,count,access=mmap.ACCESS_WRITE)
        try:
            start=time.perf_counter()
            amount=os.preadv(fds[path],[buf],offset)
            if amount!=count:raise OSError("NO_GO short full expert tensor read")
            if tail_count:
                tail=os.pread(tail_fds[path],tail_count,tail_start)
                if len(tail)!=tail_count:
                    raise OSError("NO_GO short bounded EOF tail read")
                tail_bytes+=len(tail);tails+=1
            ms.append((time.perf_counter()-start)*1000)
            bytes_read+=amount
        finally:
            buf.close()
    return {"ms":(time.perf_counter()-begun)*1000,
            "tensor_ms":ms,"direct_bytes":bytes_read,
            "buffered_tail_bytes":tail_bytes,"buffered_tail_reads":tails}

def get_event_plans(events,freq,cooc,adaptive,calibration):
    limits=Limits(session="physical-six-tensor-shadow",pin=base.PIN,
       expected_pin=base.PIN,generation=1,expected_generation=1,
       enabled=True,max_candidates=2,nvme_budget=2*LOGICAL_EXPERT_BYTES,
       gpu_budget=2*LOGICAL_EXPERT_BYTES,gpu_headroom=2*LOGICAL_EXPERT_BYTES,
       deadline_ms=20,nvme_GBps=11.319,upload_GBps=8.0,overhead_ms=0.5)
    history={}
    plans=[]
    utility=0
    candidates=0
    cold_good=cold_total=0
    for event in events:
        ranked=base.build_policy("fusion",event,history,freq,cooc)
        proposed,reason=plan(ranked,event,limits)
        if reason not in ("proposed","outside_envelope"):
            raise ValueError(f"unexpected planner status {reason}")
        if adaptive and calibration and len(proposed)>1 and not calibration.permitted(event["layer"],1):
            proposed=proposed[:1]
        picked={p["expert"] for p in proposed}
        true=set(event["ids"])
        utility+=len(true & picked)
        candidates+=len(proposed)
        cold={eid for eid,res in zip(event["ids"],event["resident"]) if res==0}
        cold_total+=len(cold)
        cold_good+=len(cold&picked)
        plans.append((event["layer"],proposed))
        history[("token",event["layer"],event["row"])]=event
        history[("call",event["call"],event["row"])]=event
    return plans,dict(candidates=candidates,precision=round(utility/max(1,candidates),5),
                      selected_recall=round(utility/(8*len(events)),5),
                      cold_coverage=round(cold_good/max(1,cold_total),5))

def replay(plans,full_index):
    base.safety()
    required={full_index[(layer,p["expert"])][j][0]
              for layer,proposed in plans for p in proposed
              if p["tier"]=="unknown_disk_or_page_cache"
              for j in range(6)}
    fds={}
    tail_fds={}
    timings=[]
    tensor_latencies=[]
    bytes_read=0
    tail_bytes=0
    tail_reads=0
    completed=0
    steps=[]
    try:
        fds={path:os.open(path,os.O_RDONLY|os.O_DIRECT|os.O_CLOEXEC) for path in required}
        tail_required={full_index[(layer,p["expert"])][j][0]
              for layer,proposed in plans for p in proposed
              if p["tier"]=="unknown_disk_or_page_cache"
              for j in range(6) if full_index[(layer,p["expert"])][j][4]}
        tail_fds={path:os.open(path,os.O_RDONLY|os.O_CLOEXEC) for path in tail_required}
        before=disk_sector_bytes()
        with BoundedShadowReader(
            session="physical-six-tensor-shadow",pin=base.PIN,generation=1,
            read_fn=lambda fd,key:physically_read_expert(fd,key,full_index,fds,tail_fds),
            max_workers=2,bytes_per_read=LOGICAL_EXPERT_BYTES,
            max_bytes=2*LOGICAL_EXPERT_BYTES) as executor:
            for n,(layer,predictions) in enumerate(plans):
                reqs=[(0,(layer,p["expert"])) for p in predictions
                      if p["tier"]=="unknown_disk_or_page_cache"]
                start=time.perf_counter()
                results=executor.read_batch(session="physical-six-tensor-shadow",
                    pin=base.PIN,generation=1,requests=reqs)
                if results:
                    steps.append((time.perf_counter()-start)*1000)
                for result in results:
                    timings.append(result["ms"])
                    tensor_latencies.extend(result["tensor_ms"])
                    bytes_read+=result["direct_bytes"]
                    tail_bytes+=result["buffered_tail_bytes"]
                    tail_reads+=result["buffered_tail_reads"]
                completed+=len(results)
                if n%16==0:base.safety()
            if completed!=executor.completed or completed!=executor.submitted:
                raise RuntimeError("NO_GO incomplete expert reads")
        sector_bytes=disk_sector_bytes()-before
    finally:
        for fd in fds.values():os.close(fd)
        for fd in tail_fds.values():os.close(fd)
    return {"full_experts_read":completed,"physical_O_DIRECT_GB":round(bytes_read/1e9,4),
            "buffered_EOF_tail_bytes":tail_bytes,"buffered_EOF_tail_reads":tail_reads,
            "physical_sector_delta_GB":round(sector_bytes/1e9,4),
            "expert_read_ms_p50":base.percentile(timings,.5),
            "expert_read_ms_p95":base.percentile(timings,.95),
            "expert_read_ms_p99":base.percentile(timings,.99),
            "tensor_read_ms_p99":base.percentile(tensor_latencies,.99),
            "two_expert_step_ms_p50":base.percentile(steps,.5),
            "two_expert_step_ms_p99":base.percentile(steps,.99)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rows",type=int,default=64)
    ap.add_argument("--execute-io",action="store_true")
    ap.add_argument("--reverse",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if not 16<=args.rows<=462:ap.error("rows 16..462")
    base.safety()
    experts=load_index()
    names=sorted(base.PRIVATE.glob("glm53_native_expert_trace_20261016_run*.jsonl"))
    if len(names)!=6 or "run05" not in names[4].name or "run06_v2" not in names[5].name:
        raise RuntimeError("NO_GO independent train/validation/evaluation mismatch")
    freq,cooc=compile_stats([read_trace(p) for p in names[:4]])
    calibration=fit(read_trace(names[4]),
                    lambda e,h:base.build_policy("fusion",e,h,freq,cooc),threshold=.4)
    heldout=read_trace(names[5])[:args.rows]
    if any(ev["v"]!=2 for ev in heldout):
        raise RuntimeError("NO_GO no full residency")
    name_order=("validation_gated","baseline") if args.reverse else ("baseline","validation_gated")
    report=[]
    for mode in name_order:
        plans,accuracy=get_event_plans(heldout,freq,cooc,mode=="validation_gated",calibration)
        if any((l,p["expert"]) not in experts for l,items in plans for p in items):
            raise RuntimeError("NO_GO planned unindexed expert")
        measured=replay(plans,experts) if args.execute_io else None
        report.append({"mode":mode,**accuracy,"physical":measured})
        base.safety()
    out={"schema":1,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "source_pin":base.PIN,"physical_model_shards":62,"full_expert_tensors_indexed":len(experts),
         "tensor_sizes_bytes":dict(TENSORS),"full_expert_logical_bytes":LOGICAL_EXPERT_BYTES,
         "train_runs":[p.name for p in names[:4]],"validation_run":names[4].name,
         "validation_run_sha256":hashlib.sha256(names[4].read_bytes()).hexdigest(),
         "evaluation_run":names[5].name,
         "evaluation_run_sha256":hashlib.sha256(names[5].read_bytes()).hexdigest(),
         "events":len(heldout),"policy_order":list(name_order),
         "candidate_second_threshold":calibration.threshold,"source_validated":True,
         "max_concurrent_full_expert_reads":2,
         "max_parallel_expert_memory_reserved":2*LOGICAL_EXPERT_BYTES,
         "physical_reads_enabled":args.execute_io,
         "results":report,
         "limitations":["Six tensor full expert O_DIRECT reads are not uploaded to Vulkan",
            "No native attention layer timestamp, no inference overlap, no tokens/sec gain",
            "Synthetic-token GLM heldout may not generalize; no native logits parity",
            "Physical block-device sector counts include concurrent traffic",
            "Consumer performance is NVMe-only and cannot prove viable 20ms deadline",
            "Some file-ending tensors require a bounded <4KiB read-only buffered EOF tail"]}
    payload=json.dumps(out,indent=2)+"\n"
    if args.out:args.out.write_text(payload)
    print(payload)
if __name__=="__main__":
    try:main()
    except Exception as error:
        print("NO_GO:",type(error).__name__,str(error),file=sys.stderr)
        raise SystemExit(2)
