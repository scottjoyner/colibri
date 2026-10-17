#!/usr/bin/env python3
"""Causal GLM-5.3 expert head v2: 4 cross-layer taps + prior same-layer token.

The previous-token route is a causal feature, not an oracle. At inference time
it is already known from an earlier verified native forward.  Uses an 8-expert
exact reuse foundation, with up to 8 extra candidates from the trained head.
No model or I/O dispatch; all evaluation is offline on independent runs.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,pathlib,sys
from replay_expert_prefetch import read_trace
from train_expert_attention import validate_splits

EXP=288
LAYERS=46
CROSS=4
WIDTH=CROSS+1
FEATURES=16

def prepare(paths):
    result=[]
    for file in paths:
        events=read_trace(file)
        by_row=collections.defaultdict(list)
        for e in events:by_row[e["row"]].append(e)
        for row,evs in by_row.items():
            past=[]
            same={}
            for ev in evs:
                context=[]
                for i in range(1,min(len(past),CROSS)+1):
                    prev=past[-i]
                    if prev["call"] != ev["call"]-i or prev["layer"]!=ev["layer"]-i:
                        break
                    context.insert(0,(prev,1))  # previous layers
                prevtoken=same.get(ev["layer"])
                if prevtoken is not None and prevtoken["call"]<ev["call"]:
                    context.append((prevtoken,2))  # same layer, earlier verified token
                if context:
                    pad=WIDTH-len(context)
                    result.append({
                        "layer":ev["layer"],"row":ev["row"],
                        "sketches":[[0.]*FEATURES for _ in range(pad)]+[e["sketch"] for e,_ in context],
                        "ids":[[0]*8 for _ in range(pad)]+[e["ids"] for e,_ in context],
                        "layers":[0]*pad+[e["layer"] for e,_ in context],
                        "types":[0]*pad+[t for _,t in context],
                        "mask":[False]*pad+[True]*len(context),
                        "prior_same":prevtoken["ids"] if prevtoken else [],
                        "target":ev["ids"]})
                past.append(ev)
                same[ev["layer"]]=ev
    return result

def factory():
    import torch
    from torch import nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.e=nn.Embedding(EXP,32)
            self.l=nn.Embedding(LAYERS,32)
            self.typ=nn.Embedding(3,32)
            self.features=nn.Linear(FEATURES,32)
            self.query=nn.Embedding(LAYERS,32)
            self.attn=nn.MultiheadAttention(32,4,batch_first=True)
            self.out=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),
                                   nn.GELU(),nn.Linear(64,EXP))
        def forward(self,sketches,ids,layers,kinds,mask,target_layer):
            # Only earlier verified native experts, never current labels.
            src=self.features(sketches)+self.e(ids).mean(dim=2)+self.l(layers)+self.typ(kinds)
            query=self.query(target_layer).unsqueeze(1)
            x,_=self.attn(query,src,src,key_padding_mask=~mask)
            return self.out(x.squeeze(1))
    return Head

def train(train_files,eval_files,out,epochs):
    import torch
    from torch.nn import functional as F
    stats=validate_splits(train_files,eval_files)
    train_rows=prepare(train_files)
    holdout=prepare(eval_files)
    if stats["train_runs"]<3 or len(train_rows)<2000 or not holdout:
        raise ValueError("NO-GO: 3 independent training runs / 2000 pairs / separate holdout required")
    freq=collections.defaultdict(collections.Counter)
    for file in train_files:
        for row in read_trace(file):
            freq[row["layer"]].update(row["ids"])
    torch.manual_seed(20261016)
    net=factory()().to("cpu")
    optimizer=torch.optim.AdamW(net.parameters(),lr=4e-4)
    def tensors(group):
        arr=lambda name,dtype: torch.tensor([r[name] for r in group],dtype=dtype)
        label=torch.zeros((len(group),EXP),dtype=torch.float32)
        for j,row in enumerate(group):label[j,row["target"]]=1
        return (arr("sketches",torch.float32),arr("ids",torch.long),
                arr("layers",torch.long),arr("types",torch.long),
                arr("mask",torch.bool),arr("layer",torch.long),label)
    for epoch in range(epochs):
        net.train()
        # deterministic but shuffled epochs, no holdout labels for optimization
        order=torch.randperm(len(train_rows),generator=torch.Generator().manual_seed(20261016+epoch)).tolist()
        for offset in range(0,len(order),64):
            batch=tensors([train_rows[j] for j in order[offset:offset+64]])
            pred=net(*batch[:-1])
            loss=F.binary_cross_entropy_with_logits(pred,batch[-1],pos_weight=torch.full((EXP,),16.))
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
    totals=collections.Counter()
    net.eval()
    with torch.no_grad():
        for offset in range(0,len(holdout),64):
            group=holdout[offset:offset+64]
            batch=tensors(group)
            predictions=net(*batch[:-1]).cpu().tolist()
            for ex,logits in zip(group,predictions):
                true=set(ex["target"])
                freq_order=[eid for eid,_ in freq[ex["layer"]].most_common()]
                prior=ex["prior_same"]
                ml=sorted(range(EXP),key=lambda e:(-logits[e],e))
                pure=ml[:16]
                # Preserve the 8 native previous-token experts, then 8 extra
                # learned candidates. Never replace verified native routing.
                hybrid=list(dict.fromkeys(prior+ml))[:16]
                baseline=list(dict.fromkeys(prior+freq_order+list(range(EXP))))[:16]
                totals["selected"]+=8
                totals["pure"]+=len(true.intersection(pure))
                totals["hybrid"]+=len(true.intersection(hybrid))
                totals["baseline"]+=len(true.intersection(baseline))
                totals["rows"]+=1
                totals["prior_available"]+=bool(prior)
    report={**stats,"schema":2,"head":"cross-layer and previous-token attention","params":sum(p.numel() for p in net.parameters()),
            "train_examples":len(train_rows),"eval_examples":len(holdout),"epochs":epochs,
            "prior_token_available_ratio":round(totals["prior_available"]/totals["rows"],5),
            "pure_head_recall_at16":round(totals["pure"]/totals["selected"],5),
            "hybrid_recall_at16":round(totals["hybrid"]/totals["selected"],5),
            "matched_fallback_recall_at16":round(totals["baseline"]/totals["selected"],5),
            "dispatch_enabled":False,"native_router_authoritative":True}
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"schema":2,"weights":net.state_dict(),"provenance":report},out)
    out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",nargs="+",required=True,type=pathlib.Path)
    p.add_argument("--eval",nargs="+",required=True,type=pathlib.Path)
    p.add_argument("--epochs",type=int,default=16)
    p.add_argument("--out",required=True,type=pathlib.Path)
    a=p.parse_args()
    if not 1<=a.epochs<=100:raise SystemExit("--epochs must be 1..100")
    try:print(json.dumps(train(a.train,a.eval,a.out,a.epochs),indent=2))
    except (ValueError,ImportError,OSError,json.JSONDecodeError) as e:
        print(f"NO-GO: {e}",file=sys.stderr);return 2
    return 0
if __name__=="__main__":raise SystemExit(main())
