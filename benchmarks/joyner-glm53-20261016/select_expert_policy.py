#!/usr/bin/env python3
"""Train-only causal fusion of three tiny GLM expert predictors.

Fit historical counts on independent train runs, select mixture weights only
on a separate validation run, then evaluate blind on later held-out runs.
The current event's route labels must never influence its own score.
No inference, prefetch, transfer, server integration, or writes outside --out.
"""
from __future__ import annotations
import argparse, collections, hashlib, itertools, json, pathlib, sys
from replay_expert_prefetch import N_EXPERTS, compile_stats, read_trace

WEIGHTS=(0.,.25,.5,1.,2.,4.)
K=(4,8,16)
def scores(e,history,freq,cooc,weights):
    w_same,w_cross,w_freq=weights
    layer=e["layer"]
    score=[0.]*N_EXPERTS
    same=history.get(("token",layer,e["row"]))
    if same:
        for position,expert in enumerate(same["ids"]):
            score[expert] += w_same*(8-position)/8
    x=history.get(("call",e["call"]-1,e["row"]))
    if x and x["layer"]==layer-1:
        for src in x["ids"]:
            outgoing=cooc.get((layer,src))
            if outgoing:
                denom=sum(outgoing.values())
                for eid,n in outgoing.items():
                    score[eid]+=w_cross*n/(8*denom)
    counts=freq.get(layer)
    if counts:
        denom=sum(counts.values())
        for eid,n in counts.items():score[eid]+=w_freq*n/denom
    return sorted(range(N_EXPERTS),key=lambda i:(-score[i],i))

def evaluate(events,freq,cooc,weights):
    history={}
    matches=collections.Counter()
    for e in events:
        ranked=scores(e,history,freq,cooc,weights)
        assert len(ranked)==N_EXPERTS and len(set(ranked))==N_EXPERTS
        target=set(e["ids"])
        for k in K:
            matches[k]+=len(target.intersection(ranked[:k]))
        matches["rows"]+=1
        history[("token",e["layer"],e["row"])]=e
        history[("call",e["call"],e["row"])]=e
    return {f"recall_at_{k}":round(matches[k]/(8*matches["rows"]),5) for k in K}

def run(train,validation,tests):
    allpaths=[*train,validation,*tests]
    if len(train)<2 or len(set(p.resolve() for p in allpaths)) != len(allpaths):
        raise ValueError("require >=2 training runs and nonoverlapping validation/test files")
    traces=[read_trace(p) for p in train]
    dev=read_trace(validation)
    freq,cooc=compile_stats(traces)
    grid=[]
    for weights in itertools.product(WEIGHTS,repeat=3):
        if not any(weights):continue
        dev_r=evaluate(dev,freq,cooc,weights)
        grid.append((dev_r["recall_at_16"],dev_r["recall_at_8"],sum(weights),weights))
    # Deterministic dev-only choice, tie breaks against large/complex weights.
    best=max(grid,key=lambda x:(x[0],x[1],-x[2],tuple(-v for v in x[3])))
    # Retrain historical frequencies on complete train+validation, never tests.
    complete_freq,complete_cooc=compile_stats([*traces,dev])
    outcomes=[]
    for path in tests:
        e=read_trace(path)
        tested=evaluate(e,complete_freq,complete_cooc,best[3])
        tested.update({"run":path.name,"trace_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                       "rows":len(e)})
        outcomes.append(tested)
    return {"schema":1,"trained_runs":[p.name for p in train],"validation_run":validation.name,
        "validation_sha256":hashlib.sha256(validation.read_bytes()).hexdigest(),
        "heldout_runs":outcomes,"dev_only_weights":{"same_layer_previous_token":best[3][0],
          "cross_layer_transition":best[3][1],"layer_frequency":best[3][2]},
        "validation_recall_at_16":best[0],"validation_recall_at_8":best[1],
        "grid_size":len(grid),"transfer_enabled":False,
        "note":"History-only prediction with frozen dev-tuned weights. No SSD or GPU transfers."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=pathlib.Path,nargs="+",required=True)
    p.add_argument("--validation",type=pathlib.Path,required=True)
    p.add_argument("--test",type=pathlib.Path,nargs="+",required=True)
    p.add_argument("--out",type=pathlib.Path)
    a=p.parse_args()
    try: report=run(a.train,a.validation,a.test)
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:
        print(f"NO-GO: {e}",file=sys.stderr);return 2
    if a.out:a.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
