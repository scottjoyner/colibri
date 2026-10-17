#!/usr/bin/env python3
"""Evaluate frozen causal attention head on a NEW independent trace (no training)."""
from __future__ import annotations
import argparse,collections,hashlib,json,pathlib,sys
from replay_expert_prefetch import read_trace
from train_expert_attention_v2 import prepare,factory

def main():
    import torch
    a=argparse.ArgumentParser()
    a.add_argument("--model",required=True,type=pathlib.Path)
    a.add_argument("--train",required=True,nargs="+",type=pathlib.Path)
    a.add_argument("--eval",required=True,type=pathlib.Path)
    a.add_argument("--out",type=pathlib.Path)
    args=a.parse_args()
    try:
        blob=torch.load(args.model,map_location="cpu",weights_only=True)
        if blob.get("schema")!=2:raise ValueError("wrong model schema")
        pinned=blob["provenance"]["train_trace_sha256"]
        if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.train}!=pinned:
            raise ValueError("different training traces than the pinned weights")
        if args.eval.name in pinned:
            raise ValueError("evaluation trace also used for training")
        eval_hash=hashlib.sha256(args.eval.read_bytes()).hexdigest()
        if eval_hash in blob["provenance"]["eval_trace_sha256"].values():
            raise ValueError("evaluation run was used for model selection; require fresh run")
        freq=collections.defaultdict(collections.Counter)
        for tr in args.train:
            for row in read_trace(tr):
                freq[row["layer"]].update(row["ids"])
        data=prepare([args.eval])
        if len(data)<1:raise ValueError("empty evaluation")
        net=factory()().to("cpu")
        net.load_state_dict(blob["weights"])
        net.eval()
        matches=collections.Counter()
        with torch.no_grad():
            for k in range(0,len(data),64):
                group=data[k:k+64]
                inp=lambda key,dtype:torch.tensor([row[key] for row in group],dtype=dtype)
                logits=net(inp("sketches",torch.float32),inp("ids",torch.long),
                           inp("layers",torch.long),inp("types",torch.long),
                           inp("mask",torch.bool),inp("layer",torch.long)).tolist()
                for row,score in zip(group,logits):
                    native=set(row["target"])
                    ranked=sorted(range(288),key=lambda eid:(-score[eid],eid))
                    prior=row["prior_same"]
                    baseline=list(dict.fromkeys(prior+[eid for eid,_ in freq[row["layer"]].most_common()]+list(range(288))))[:16]
                    hybrid=list(dict.fromkeys(prior+ranked))[:16]
                    for key,ids in (("pure",ranked[:16]),("hybrid",hybrid),("matched_baseline",baseline)):
                        matches[key]+=len(native.intersection(ids))
                    matches["total"]+=len(native)
                    matches["rows"]+=1
        report={"schema":2,"model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "eval_trace_sha256":eval_hash,"heldout_rows":matches["rows"],
               "native_selection_count":matches["total"],
               "pure_recall_at16":round(matches["pure"]/matches["total"],5),
               "hybrid_recall_at16":round(matches["hybrid"]/matches["total"],5),
               "matched_prior_frequency_recall_at16":round(matches["matched_baseline"]/matches["total"],5),
               "training_runs":len(args.train),"dispatch_enabled":False}
        if args.out:args.out.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
        return 0
    except (ValueError,RuntimeError,OSError,KeyError) as e:
        print(f"NO-GO: {e}",file=sys.stderr)
        return 2

if __name__=="__main__":raise SystemExit(main())
