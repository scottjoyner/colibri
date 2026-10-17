#!/usr/bin/env python3
"""Causal shadow of GLM-5.3 Vulkan expert residency; NOT native throughput.

Reads separate native validation and frozen holdout routing traces; never routes
requests, reads tensor bytes, or changes resident caches on the machine.
GPU expert upload is modeled as completing *after the forward*, never during
that forward; ideal hit counts do not prove speedup or exact vkt semantics.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

LAYERS=42
EXPERT_GIB=13.5/1024
VALIDATION_NAME="glm53_native_expert_trace_20261016_run05.jsonl"
HOLDOUT_NAME="glm53_native_expert_trace_20261016_run06_v2.jsonl"

def load_forwards(path: Path):
    raw=path.read_bytes()
    groups=defaultdict(list)
    for line in raw.splitlines():
        if not line.strip():continue
        e=json.loads(line)
        if not (0<=e["layer"]<100 and 0<=e["call"]<1000000 and
                1<=e["topk"]<=16 and len(e["ids"])==e["topk"] and
                all(0<=x<e["n_experts"] for x in e["ids"])):
            raise ValueError("NO_GO malformed routing record")
        groups[e["call"]].append(e)
    if len(groups)%LAYERS:
        raise ValueError("NO_GO incomplete forward frame")
    calls=sorted(groups)
    if calls != list(range(len(groups))):
        raise ValueError("NO_GO missing/duplicate forward calls")
    forwards=[]
    for at in range(0,len(calls),LAYERS):
        selected=[]
        for call in calls[at:at+LAYERS]:
            per=groups[call]
            if len(set(e["layer"] for e in per))!=1:
                raise ValueError("NO_GO call mixes routing layers")
            for e in per:
                selected.extend((e["layer"],int(i)) for i in e["ids"])
        forwards.append(selected)
    return forwards,hashlib.sha256(raw).hexdigest()

def simulate(frames, capacity:int, rate:int, policy:str) -> dict:
    if capacity<1 or rate<0 or policy not in ("first","repeat_priority","repeat_gate","balanced"):
        raise ValueError("NO_GO parameter")
    resident: OrderedDict[tuple[int,int],None]=OrderedDict()
    seen=Counter()
    hits=0;selections=0;upload=0;evict=0;eligible=0;rejected_first=0
    for frame in frames:
        candidates={}
        group_size=max(1,len(frame)//LAYERS) # prefill has 3 rows
        for key in frame:
            selections+=1
            if key in resident:
                hits+=1
                resident.move_to_end(key)
            else:
                # First occurrence order only; no same-forward GPU read-back.
                if key not in candidates:candidates[key]=len(candidates)
        # Observe a given expert at most once per forward, regardless of
        # whether a prompt prefill has many rows for that layer.
        for key in set(frame):
            seen[key]+=1
        # Admit AFTER every forward. Equivalent to a zero-latency model for
        # subsequent tokens, optimistic compared with real upload fences.
        keys=list(candidates)
        eligible+=len(keys)
        if policy in ("repeat_priority","repeat_gate"):
            keys.sort(key=lambda key:(-seen[key],candidates[key]))
        elif policy=="balanced":
            count=Counter(layer for layer,_ in resident)
            keys.sort(key=lambda key:(count[key[0]],-seen[key],candidates[key]))
        if policy=="repeat_gate":
            allowed=[key for key in keys if seen[key]>=2]
            rejected_first+=len(keys)-len(allowed)
            keys=allowed
        # prefill can accrue 3 tokens' worth of rate in this first forward
        step_cap=rate*(3 if group_size>1 else 1)
        for key in keys[:step_cap]:
            if key in resident:continue
            if len(resident)>=capacity:
                resident.popitem(last=False);evict+=1
            resident[key]=None
            upload+=1
    return dict(selected=selections,hits=hits,hit_fraction=round(hits/selections,5),
                uploads=upload,evictions=evict,
                opportunity=eligible,repeat_first_refused=rejected_first,
                remaining_resident=len(resident),
                # minimum bytes sent to GPU if all uploads succeed,
                # ignoring pool fragmentation and staging replication.
                upload_lower_bound_gib=round(upload*EXPERT_GIB,3))

def analyze(paths:Path, *, explicit_validation:Path|None=None,
            explicit_holdout:Path|None=None):
    val_file=explicit_validation or paths/VALIDATION_NAME
    hold_file=explicit_holdout or paths/HOLDOUT_NAME
    validation,valsha=load_forwards(val_file)
    hold,holdsha=load_forwards(hold_file)
    if not 16<=len(validation)<=100 or not 8<=len(hold)<=100:
        raise ValueError("NO_GO wrong native validation/holdout length")
    candidates=[]
    for cap in (80,160,320):
        for rate in (16,32):
            for policy in ("first","repeat_priority","repeat_gate","balanced"):
                score=simulate(validation,cap,rate,policy)
                candidates.append(dict(capacity=cap,rate=rate,policy=policy,
                                       validation=score))
    # Pre-registered conservative objective: prioritize hits and penalize
    # SSD->VRAM traffic. Bound GPU expert capacity to <=320 (roughly 8 GiB),
    # rate <=32 to avoid scaling 26 GiB transfers from prior experiments.
    # Data from holdout are NEVER used for selection.
    def rank(x):
        v=x["validation"]
        return (v["hits"]-0.30*v["uploads"],v["hits"],
                -v["uploads"],-x["capacity"])
    best=max(candidates,key=rank)
    references=[
       {"capacity":80,"rate":16,"policy":"first"},
       {"capacity":80,"rate":16,"policy":"repeat_gate"},
       {"capacity":80,"rate":16,"policy":"repeat_priority"},
       {"capacity":320,"rate":32,"policy":"first"},
       {"capacity":320,"rate":32,"policy":"repeat_gate"},
       {"capacity":320,"rate":32,"policy":"repeat_priority"},
    ]
    chosen=[]
    for c in [*references,
              {key:best[key] for key in ("capacity","rate","policy")}]:
        result=simulate(hold,**c)
        chosen.append(dict(**c,holdout=result))
    def distinct(frames):
        allkeys=[k for frame in frames for k in frame]
        return {"events":sum(map(len,frames)),
                "unique_layer_experts":len(set(allkeys)),
                "repeated_selections":len(allkeys)-len(set(allkeys))}
    return {
      "schema":1,"policy":"read-only causally replayed idealized GPU residency",
      "no_future_expert_prediction":True,
      "gpu_uploads_complete_before_next_forward_assumed":True,
      "measured_native_tokens_per_second":False,
      "validation_sha256":valsha,"holdout_sha256":holdsha,
      "validation_forward_count":len(validation),
      "holdout_forward_count":len(hold),
      "validation_summary":distinct(validation),"holdout_summary":distinct(hold),
      "objective":"validation hits - 0.30 * uploads, cap <=320 experts and <=32 uploads/token",
      "best_validation_policy":{key:best[key] for key in ("capacity","rate","policy")},
      "validation_top5":sorted(candidates,key=rank,reverse=True)[:5],
      "heldout_results":chosen,
      "limits":"simulator is not native Vulkan tier: ignores admission heat gates, async upload stalls, CPU cache, pool fragmentation, and actual work latency"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trace-dir",type=Path,required=True)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    r=analyze(a.trace_dir)
    result=json.dumps(r,indent=2,sort_keys=True)+"\n"
    print(result)
    if a.out:
        if a.out.exists():raise ValueError("NO_GO output already exists")
        a.out.write_text(result)

if __name__=="__main__":
    main()
