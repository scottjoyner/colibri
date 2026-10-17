#!/usr/bin/env python3
"""Optional CPU-trainable cross-layer attention head; never dispatches I/O.

Only feature vectors and *prior layer* native routes are visible to the head.
Native routes in the target layer are training labels ONLY, never inputs.

No torch required to inspect/validate a dataset. Training refuses tiny or
non-independent datasets. GPU/network/production integrations are excluded.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib, sys
from replay_expert_prefetch import read_trace

LAYERS=46
EXPERTS=288
WINDOW=4
FEATURES=16
MIN_TRAIN_RUNS=3
MIN_TRAIN_EXAMPLES=2000

def load_sequences(paths):
    out=[]
    for path in paths:
        events=read_trace(path)
        by_row=collections.defaultdict(list)
        for e in events:
            by_row[e["row"]].append(e)
        for row,seq in by_row.items():
            hist=[]
            for e in seq:
                context=[]
                for i in range(1,min(len(hist),WINDOW)+1):
                    prev=hist[-i]
                    if prev["call"] != e["call"]-i or prev["layer"]!=e["layer"]-i:
                        break
                    context.insert(0,prev)
                if context:
                    pad=WINDOW-len(context)
                    out.append({
                        "future_layer":e["layer"],
                        "sketches":[[0.0]*FEATURES for _ in range(pad)]+[x["sketch"] for x in context],
                        "ids":[[0]*8 for _ in range(pad)]+[x["ids"] for x in context],
                        "layers":[0]*pad+[x["layer"] for x in context],
                        "mask":[False]*pad+[True]*len(context),
                        "target":e["ids"],
                    })
                hist.append(e)
    return out

def validate_splits(train,eval_):
    if not train or not eval_:raise ValueError("train and held-out eval run files required")
    tr={p.resolve() for p in train};ev={p.resolve() for p in eval_}
    if tr&ev:raise ValueError("training and evaluation run overlap")
    stats={"train_runs":len(tr),"eval_runs":len(ev),
           "train_samples":len(load_sequences(train)),
           "eval_samples":len(load_sequences(eval_)),
           "train_trace_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in train},
           "eval_trace_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in eval_}}
    return stats

def build_torch_model():
    import torch
    from torch import nn
    class ExpertPrefetchAttentionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=nn.Embedding(EXPERTS,32)
            self.layer=nn.Embedding(LAYERS,32)
            self.project=nn.Linear(FEATURES,32)
            self.attention=nn.MultiheadAttention(32,4,batch_first=True)
            self.query=nn.Embedding(LAYERS,32)
            self.output=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),
                                      nn.GELU(),nn.Linear(64,EXPERTS))
        def forward(self, sketches, ids, layers, mask, target_layer):
            features=self.project(sketches)
            inputs=features+self.embedding(ids).mean(dim=2)+self.layer(layers)
            q=self.query(target_layer).unsqueeze(1)
            # padded inputs have no influence: key_padding_mask=True means mask
            latent,_=self.attention(q,inputs,inputs,key_padding_mask=~mask)
            return self.output(latent.squeeze(1))
    return ExpertPrefetchAttentionHead

def train_head(train,eval_,out,epochs=8):
    import torch
    from torch.nn import functional as F
    stats=validate_splits(train,eval_)
    if stats["train_runs"]<MIN_TRAIN_RUNS or stats["train_samples"]<MIN_TRAIN_EXAMPLES:
        raise ValueError("not enough independent GLM traces: need >=3 train runs and >=2000 cross-layer training events")
    torch.manual_seed(20261016)
    maker=build_torch_model()
    model=maker().to("cpu")
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    train_examples=load_sequences(train)
    eval_examples=load_sequences(eval_)
    def collate(rows):
        x=lambda key,dtype:torch.tensor([r[key] for r in rows],dtype=dtype)
        target=torch.zeros((len(rows),EXPERTS))
        for i,row in enumerate(rows):target[i,row["target"]]=1.
        return (x("sketches",torch.float32),x("ids",torch.long),x("layers",torch.long),
                x("mask",torch.bool),x("future_layer",torch.long),target)
    for epoch in range(epochs):
        model.train()
        for offset in range(0,len(train_examples),64):
            batch=train_examples[offset:offset+64]
            data=collate(batch)
            preds=model(*data[:-1])
            loss=F.binary_cross_entropy_with_logits(preds,data[-1],pos_weight=torch.tensor([16.0]*EXPERTS))
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
    model.eval()
    matched=seen=0
    with torch.no_grad():
        for offset in range(0,len(eval_examples),64):
            data=collate(eval_examples[offset:offset+64])
            preds=model(*data[:-1]).topk(16,dim=1).indices
            matched+=int(data[-1].gather(1,preds).sum().item())
            seen+=len(preds)*8
    recall=matched/max(1,seen)
    # Holdout benchmark is diagnostics, not a real transfer-speed improvement.
    metrics={**stats,"head":"causal-4-layer-attention", "params":sum(p.numel() for p in model.parameters()),
             "heldout_recall_at16":round(recall,5),"epochs":epochs,
             "transfer_enabled":False,"native_router_authoritative":True}
    path=out.with_suffix(".json")
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"weights":model.state_dict(),"schema":1,"training":metrics},out)
    path.write_text(json.dumps(metrics,indent=2)+"\n")
    return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",nargs="+",type=pathlib.Path,required=True)
    ap.add_argument("--eval",nargs="+",type=pathlib.Path,required=True)
    ap.add_argument("--inspect-only",action="store_true")
    ap.add_argument("--out",type=pathlib.Path)
    ap.add_argument("--epochs",type=int,default=8)
    args=ap.parse_args()
    try:
        stats=validate_splits(args.train,args.eval)
        if args.inspect_only:
            print(json.dumps({**stats,"qualified_for_training":
                 stats["train_runs"]>=MIN_TRAIN_RUNS and stats["train_samples"]>=MIN_TRAIN_EXAMPLES},indent=2))
            return 0
        if args.out is None or not (1<=args.epochs<=100):
            raise ValueError("--out and 1<=--epochs<=100 required to train")
        results=train_head(args.train,args.eval,args.out,args.epochs)
        print(json.dumps(results,indent=2));return 0
    except (ValueError,ImportError,OSError,json.JSONDecodeError) as e:
        print(f"NO-GO: {e}",file=sys.stderr);return 2

if __name__=="__main__":raise SystemExit(main())
