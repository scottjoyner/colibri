#!/usr/bin/env python3
"""Frozen prefetch confidence gate calibrated ONLY on independent validation run.

Native routing ids are examined only AFTER that validation event is ranked.
Per-layer smoothed probability of the second nominated expert being useful.
It does not change native routing or access held-out labels during planning.
"""
from __future__ import annotations
import collections
from dataclasses import dataclass

@dataclass(frozen=True)
class Calibration:
    layer_second_probability: dict[int,float]
    global_second_probability: float
    train_rows: int
    prior_strength: float
    threshold: float
    def permitted(self,layer:int,position:int)->bool:
        if position==0:return True
        if position!=1:return False
        return self.layer_second_probability.get(layer,self.global_second_probability)>=self.threshold

def fit(validation_events,rank_func,threshold:float=0.40,prior_strength:float=8.0)->Calibration:
    if not 0<=threshold<=1 or prior_strength<=0:raise ValueError("invalid calibration knobs")
    hits=collections.Counter();counts=collections.Counter();history={}
    total=0;matched=0
    for ev in validation_events:
        # Freeze rank BEFORE observing current truth; past events alone in history.
        rank=rank_func(ev,history)
        if len(rank)<2 or rank[0]==rank[1]:raise ValueError("invalid native-independent ranking")
        positive=int(rank[1] in ev["ids"])
        hits[ev["layer"]]+=positive
        counts[ev["layer"]]+=1
        matched+=positive;total+=1
        history[("token",ev["layer"],ev["row"])]=ev
        history[("call",ev["call"],ev["row"])]=ev
    if total<20:raise ValueError("validation too small")
    global_p=matched/total
    post={l:(hits[l]+prior_strength*global_p)/(counts[l]+prior_strength)
          for l in counts}
    return Calibration(post,global_p,total,prior_strength,threshold)
