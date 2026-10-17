#!/usr/bin/env python3
"""SHADOW only: bound next-expert nominations by cache tier, byte quotas and deadline."""
from __future__ import annotations
import argparse, collections, dataclasses, hashlib, json, math, pathlib, sys
from replay_expert_prefetch import N_EXPERTS, BYTES_PER_EXPERT, compile_stats, read_trace
from select_expert_policy import scores

@dataclasses.dataclass(frozen=True)
class Limits:
    session: str
    pin: str
    expected_pin: str
    generation: int
    expected_generation: int
    enabled: bool = False
    max_candidates: int = 2
    nvme_budget: int = 2*BYTES_PER_EXPERT
    gpu_budget: int = 2*BYTES_PER_EXPERT
    gpu_headroom: int = 2*BYTES_PER_EXPERT
    deadline_ms: float = 0.0
    nvme_GBps: float = 0.
    upload_GBps: float = 0.
    overhead_ms: float = 0.5

def in_mask(mask,e):
    return bool(mask[e>>3] & (1<<(e&7)))

def plan(rank, snapshot, limit):
    """Never looks at snapshot['ids']: the native route is an evaluation label only."""
    if not limit.enabled:return [],"disabled"
    if (len(limit.pin)!=40 or limit.pin!=limit.expected_pin or
        any(c not in "0123456789abcdef" for c in limit.pin)):
        return [],"pin_mismatch"
    if not limit.session or limit.generation<0 or limit.generation!=limit.expected_generation:
        return [],"stale_session_or_generation"
    if snapshot.get("v")!=2 or snapshot.get("n_experts")!=N_EXPERTS:
        return [],"missing_full_inventory"
    try:
        host=bytes.fromhex(snapshot["host_mask"])
        gpu=bytes.fromhex(snapshot["gpu_mask"])
    except (KeyError,ValueError,TypeError):return [],"bad_cache_mask"
    if len(host)!=36 or len(gpu)!=36:return [],"bad_cache_mask"
    resources=(limit.deadline_ms,limit.nvme_GBps,limit.upload_GBps,limit.overhead_ms)
    if (not all(math.isfinite(v) for v in resources) or
        min(resources[:3])<=0 or limit.overhead_ms<0 or
        not 1<=limit.max_candidates<=8 or
        min(limit.nvme_budget,limit.gpu_budget,limit.gpu_headroom)<0):
        return [],"no_valid_resource_envelope"
    if (not isinstance(rank,list) or len(rank)!=len(set(rank)) or
        any(type(e) is not int or not 0<=e<N_EXPERTS for e in rank)):
        return [],"invalid_candidates"
    proposed=[];nvme=0;gpu_bytes=0;elapsed=0.
    for eid in rank:
        if in_mask(gpu,eid):continue
        cold=not in_mask(host,eid)
        new_nvme=BYTES_PER_EXPERT if cold else 0
        add_ms=(new_nvme/(limit.nvme_GBps*1e6) +
                BYTES_PER_EXPERT/(limit.upload_GBps*1e6) + limit.overhead_ms)
        if (len(proposed)>=limit.max_candidates or
            nvme+new_nvme>limit.nvme_budget or
            gpu_bytes+BYTES_PER_EXPERT>min(limit.gpu_budget,limit.gpu_headroom) or
            elapsed+add_ms>limit.deadline_ms):
            continue
        proposed.append({"expert":eid,"tier":"unknown_disk_or_page_cache" if cold else "host_ram",
                         "nominal_nvme_bytes":new_nvme,"nominal_gpu_bytes":BYTES_PER_EXPERT,
                         "assumed_ms":round(add_ms,3)})
        nvme+=new_nvme;gpu_bytes+=BYTES_PER_EXPERT;elapsed+=add_ms
    return proposed,"proposed" if proposed else "outside_envelope"

def simulate(train_files,heldout,weights,limits):
    if heldout.resolve() in {x.resolve() for x in train_files}:raise ValueError("train/eval overlap")
    freq,cooc=compile_stats([read_trace(f) for f in train_files])
    test=read_trace(heldout)
    if any(x["v"]!=2 for x in test):raise ValueError("v2 full inventory required")
    history={};count=collections.Counter()
    for event in test:
        ranked=scores(event,history,freq,cooc,weights)
        candidates,status=plan(ranked,event,limits)
        nominated={e["expert"] for e in candidates}
        selected=set(event["ids"])  # OFFLINE evaluation only; NEVER passed to plan
        cold={e for e,res in zip(event["ids"],event["resident"]) if res==0}
        count["rows"]+=1;count["native_selected"]+=len(selected)
        count["native_cold"]+=len(cold)
        count["nominated"]+=len(candidates)
        count["correct"]+=len(nominated & selected)
        count["correct_cold"]+=len(nominated & cold)
        count["unneeded"]+=len(nominated-selected)
        count["candidate_nvme_bytes"]+=sum(x["nominal_nvme_bytes"] for x in candidates)
        count["candidate_gpu_bytes"]+=sum(x["nominal_gpu_bytes"] for x in candidates)
        count["gate_"+status]+=1
        history[("token",event["layer"],event["row"])]=event
        history[("call",event["call"],event["row"])]=event
    return {"schema":1,"shadow_only":True,"selected_recall":round(count["correct"]/count["native_selected"],5),
            "native_cold_coverage_upper":round(count["correct_cold"]/max(1,count["native_cold"]),5),
            "candidate_precision":round(count["correct"]/max(1,count["nominated"]),5),
            "candidate_ssd_or_page_cache_gb":round(count["candidate_nvme_bytes"]/1e9,4),
            "candidate_gpu_upload_gb":round(count["candidate_gpu_bytes"]/1e9,4),
            "counts":dict(count),"trace_sha256":hashlib.sha256(heldout.read_bytes()).hexdigest(),
            "actual_ssd_bytes":None,"speedup":None,"source_pin":limits.pin}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",nargs="+",required=True,type=pathlib.Path)
    p.add_argument("--eval",required=True,type=pathlib.Path)
    p.add_argument("--policy",required=True,type=pathlib.Path)
    p.add_argument("--source-pin",required=True)
    p.add_argument("--nvme-gbps",type=float,required=True)
    p.add_argument("--upload-gbps",type=float,required=True)
    p.add_argument("--window-ms",type=float,default=10.)
    p.add_argument("--max-candidates",type=int,default=2)
    p.add_argument("--out",type=pathlib.Path)
    a=p.parse_args()
    try:
        params=json.loads(a.policy.read_text())
        if a.eval.name==params["validation_run"]:raise ValueError("heldout is validation run")
        w=params["dev_only_weights"]
        weights=(w["same_layer_previous_token"],w["cross_layer_transition"],w["layer_frequency"])
        limits=Limits(session="private-replay",pin=a.source_pin,expected_pin=a.source_pin,
                      generation=1,expected_generation=1,enabled=True,
                      nvme_GBps=a.nvme_gbps,upload_GBps=a.upload_gbps,
                      deadline_ms=a.window_ms,max_candidates=a.max_candidates)
        result=simulate(a.train,a.eval,weights,limits)
        result["assumptions"]={"nvme_gbps":a.nvme_gbps,"upload_gbps":a.upload_gbps,
                               "window_ms":a.window_ms,"max_candidates":a.max_candidates,
                               "note":"Upload bandwidth & deadline hypothetical, NOT hardware measured. No transfers performed."}
        text=json.dumps(result,indent=2)+"\n"
        if a.out:a.out.write_text(text)
        print(text,end="")
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:
        print("NO-GO:",e,file=sys.stderr);return 2
    return 0
if __name__=="__main__":raise SystemExit(main())
