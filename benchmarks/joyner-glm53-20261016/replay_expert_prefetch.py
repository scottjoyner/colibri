#!/usr/bin/env python3
"""Offline-only, leakage-controlled replay for GLM-5.3 expert-prefetch policies.

Input: independent explicit-opt-in GLM53_EXPERT_TRACE JSONL *run files*. Splits
must be entire runs. Never derive a future label from the same evaluation run
to train. No prefetch, GPU access, or checkpoint writes are performed here.
"""
from __future__ import annotations
import argparse, collections, json, math, pathlib, statistics, sys

N_EXPERTS=288
BYTES_PER_EXPERT=14_155_776  # first-pass planning approximation; not physical I/O evidence

def read_trace(path: pathlib.Path):
    rows=[]
    last=-1
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        e=json.loads(line)
        if (e.get("v") not in (1,2) or type(e.get("call")) is not int
            or e["call"]<0 or e["call"]<last
            or e.get("n_experts")!=N_EXPERTS
            or e.get("topk")!=8 or e.get("layer",0) < 0
            or type(e.get("row")) is not int or e["row"]<0
            or len(e.get("ids",[]))!=8 or len(set(e["ids"]))!=8
            or not all(type(i) is int and 0<=i<N_EXPERTS for i in e["ids"])
            or len(e.get("resident",[]))!=8
            or not all(i in (0,1,2) for i in e["resident"])
            or len(e.get("sketch",[]))!=16
            or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in e["sketch"])
            or len(e.get("gates",[]))!=8 or not all(isinstance(g,(float,int)) and math.isfinite(g) for g in e["gates"])):
            raise ValueError(f"{path}:{line_no}: invalid or incompatible GLM53 expert trace")
        if e["v"]==2:
            mask_bytes=(N_EXPERTS+7)//8
            try:
                host=bytes.fromhex(e["host_mask"])
                gpu=bytes.fromhex(e["gpu_mask"])
            except (KeyError,TypeError,ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: missing/invalid full residency bitmap") from exc
            if len(host)!=mask_bytes or len(gpu)!=mask_bytes:
                raise ValueError(f"{path}:{line_no}: bitmap length mismatch")
            for eid,tier in zip(e["ids"],e["resident"]):
                in_ram=bool(host[eid>>3]&(1<<(eid&7)))
                in_gpu=bool(gpu[eid>>3]&(1<<(eid&7)))
                derived=2 if in_gpu else 1 if in_ram else 0
                if derived!=tier:
                    raise ValueError(f"{path}:{line_no}: selected expert residency contradicts bitmaps")
        last=e["call"];rows.append(e)
    if not rows:raise ValueError(f"{path}: empty or missing capture")
    return rows

def compile_stats(traces):
    freq=collections.defaultdict(collections.Counter)
    cooc=collections.defaultdict(collections.Counter)
    for events in traces:
        prior={}  # within a single run only
        for ev in events:
            key=(ev["call"],ev["row"])
            for expert in ev["ids"]:freq[ev["layer"]][expert]+=1
            if ev["layer"]>0:
                # Match only immediately preceding layer within this same run
                prev=prior.get((ev["call"]-1,ev["row"]))
                if prev and prev["layer"]==ev["layer"]-1:
                    for src in prev["ids"]:
                        for tgt in ev["ids"]:cooc[(ev["layer"],src)][tgt]+=1
            prior[key]=ev
    return freq,cooc

def choose(policy,ev,prior,freq,cooc,n):
    l=ev["layer"]
    score=collections.Counter()
    if policy=="freq":
        score.update(freq[l])
    elif policy=="prior-token":
        prev=prior.get(("token",l,ev["row"]))
        if prev:score.update({ex:(8-i) for i,ex in enumerate(prev["ids"])})
        else:score.update(freq[l])
    elif policy=="cross-layer":
        prev=prior.get(("call",ev["call"]-1,ev["row"]))
        if prev and prev["layer"]==l-1:
            for src in prev["ids"]:
                score.update(cooc[(l,src)])
        if not score:score.update(freq[l])
    else:raise ValueError(policy)
    # All policies must consume the SAME candidate budget. Previous-token has
    # only 8 native IDs, so add layer-frequency fallbacks rather than allowing
    # it to win the transfer-cost comparison by emitting fewer predictions.
    ordered=[eid for eid,_ in sorted(score.items(),key=lambda kv:(-kv[1],kv[0]))]
    fallback=[eid for eid,_ in sorted(freq[l].items(),key=lambda kv:(-kv[1],kv[0]))]
    return list(dict.fromkeys(ordered+fallback+list(range(N_EXPERTS))))[:n]

def replay(train_files,eval_files,budget):
    train=[read_trace(p) for p in train_files]
    test=[read_trace(p) for p in eval_files]
    if set(map(str,train_files)) & set(map(str,eval_files)):
        raise ValueError("evaluation files cannot appear in training")
    if len(train_files)<2 or len(eval_files)<1:
        raise ValueError("requires >=2 independent training runs and >=1 held-out run")
    freq,cooc=compile_stats(train)
    out=[]
    for policy in ("freq","prior-token","cross-layer"):
        matched=actual_misses=covered_misses=prefetch_bytes=wasted_bytes=0
        exact_unknown_bytes=exact_upload_bytes=exact_wasted_bytes=inventory_rows=0
        nselected=nr=ncandidates=0
        for events in test:
            prior={}
            # Each event is evaluated BEFORE learning its native route.
            for ev in events:
                preds=choose(policy,ev,prior,freq,cooc,budget)
                if len(preds)!=budget or len(set(preds))!=budget:
                    raise ValueError("cache policy failed equal-budget invariant")
                ncandidates+=len(preds)
                target=set(ev["ids"])
                cold={eid for eid,res in zip(ev["ids"],ev["resident"]) if res==0}
                resident={eid for eid,res in zip(ev["ids"],ev["resident"]) if res!=0}
                matched+=len(target.intersection(preds))
                actual_misses+=len(cold)
                covered_misses+=len(cold.intersection(preds))
                # No global cache simulation: theoretical upper bound where
                # all predicted unknown experts transfer in time.
                sent=set(preds)-resident
                prefetch_bytes+=len(sent)*BYTES_PER_EXPERT
                wasted_bytes+=len(sent-target)*BYTES_PER_EXPERT
                if ev["v"]==2:
                    host=bytes.fromhex(ev["host_mask"])
                    gpu=bytes.fromhex(ev["gpu_mask"])
                    def has(mask,eid):return bool(mask[eid>>3]&(1<<(eid&7)))
                    # App-level tiers only. SSD page cache presence is UNKNOWN.
                    unknown={e for e in preds if not has(host,e) and not has(gpu,e)}
                    upload={e for e in preds if not has(gpu,e)}
                    inventory_rows+=1
                    exact_unknown_bytes+=len(unknown)*BYTES_PER_EXPERT
                    exact_upload_bytes+=len(upload)*BYTES_PER_EXPERT
                    exact_wasted_bytes+=len(unknown-target)*BYTES_PER_EXPERT
                nselected+=len(target);nr+=1
                prior[("call",ev["call"],ev["row"])]=ev
                prior[("token",ev["layer"],ev["row"])]=ev
        out.append({"policy":policy,"budget_experts_per_step":budget,"rows":nr,
            "candidate_slots":ncandidates,
            "full_inventory_rows":inventory_rows,
            "known_missing_host_and_gpu_gb":round(exact_unknown_bytes/1e9,4) if inventory_rows==nr else None,
            "known_missing_gpu_gb":round(exact_upload_bytes/1e9,4) if inventory_rows==nr else None,
            "known_missing_wasted_gb":round(exact_wasted_bytes/1e9,4) if inventory_rows==nr else None,
            "selected_recall":round(matched/nselected,5),
            "cold_coverage_upper_bound":round(covered_misses/actual_misses,5) if actual_misses else None,
            "estimated_prefetch_gb":round(prefetch_bytes/1e9,4),
            "estimated_wasted_gb":round(wasted_bytes/1e9,4)})
    return {"schema":1,"split":"independent_run_holdout","train_files":[p.name for p in train_files],
            "evaluation_files":[p.name for p in eval_files],
            "note":"Counterfactual only; no speculative transfers issued. v2 traces contain full known RAM/GPU bitmaps but cannot tell whether Linux page cache contains an expert. Physical SSD bandwidth/speedup unmeasured.",
            "rows_evaluated":sum(len(ev) for ev in test),"policies":out}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",nargs="+",type=pathlib.Path,required=True)
    ap.add_argument("--eval",nargs="+",type=pathlib.Path,required=True)
    ap.add_argument("--budget",type=int,choices=[1,2,4,8,16,24,32],default=16)
    ap.add_argument("--out",type=pathlib.Path)
    a=ap.parse_args()
    try: result=replay(a.train,a.eval,a.budget)
    except (ValueError,OSError,json.JSONDecodeError) as ex:
        print(f"NO-GO: {ex}",file=sys.stderr);return 2
    data=json.dumps(result,indent=2)+"\n"
    if a.out: a.out.write_text(data)
    print(data,end="")
    return 0

if __name__=="__main__":raise SystemExit(main())
