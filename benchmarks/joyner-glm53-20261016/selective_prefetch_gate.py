#!/usr/bin/env python3
"""Read-only 24/7-safe gate for optional GLM53 expert prefetch nominations.

This module never performs I/O or GPU operations. It cannot authorize
fleet production or call the native inference engine. Admission is per
observation and fails closed if a current hardware capacity witness, exact
model/session lease, independently calibrated utility, or time budget is
missing. It NEVER evicts GPU residents to make room for speculative experts.

Threshold and performance measurements must come from a disjoint validation
run; a caller cannot substitute the current true routing IDs for confidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping

from replay_expert_prefetch import BYTES_PER_EXPERT, N_EXPERTS
from shadow_tier_plan import Limits, plan

MIB=1024*1024
MAX_AGE_MS=1500
MIN_CONFIDENCE=.70
MIN_USEFUL_SAMPLES=30
MIN_VRAM_MARGIN=1*1024**3
MAX_PER_EVENT_BYTES=2*BYTES_PER_EXPERT

@dataclass(frozen=True)
class Witness:
    model_pin:str
    session:str
    generation:int
    observed_at_ms:int
    current_time_ms:int
    available_vram_bytes:int
    owner_resident_bytes:int
    reserved_for_service_bytes:int
    speculative_bytes_inflight:int
    queue_depth:int
    max_queue_depth:int
    gpu_healthy:bool
    ssd_healthy:bool
    memory_psi_full_avg10:float
    source:str="read_only_native_counters"

@dataclass(frozen=True)
class Confidence:
    # Validated on independent, earlier data. Scores represent probability
    # a nominated expert is accessed before its useful deadline.
    model_pin:str
    source_runs:int
    validation_rows:int
    verified_split:bool
    predicted_probability:Mapping[int,float]
    validation_precision:float

def candidate(snapshot:dict,rank:list[int],limit:Limits,
              witness:Witness,confidence:Confidence,
              *,minimum_confidence:float=MIN_CONFIDENCE) -> tuple[list[dict],str]:
    if not limit.enabled:
        return [],"disabled"
    if not (witness.source=="read_only_native_counters" and
            witness.model_pin==confidence.model_pin==limit.pin==limit.expected_pin and
            witness.session==limit.session and
            witness.generation==limit.generation==limit.expected_generation):
        return [],"identity_or_lease_mismatch"
    if (type(witness.current_time_ms) is not int or
        type(witness.observed_at_ms) is not int or
        witness.current_time_ms<witness.observed_at_ms or
        witness.current_time_ms-witness.observed_at_ms>MAX_AGE_MS):
        return [],"stale_resource_witness"
    if not witness.gpu_healthy or not witness.ssd_healthy:
        return [],"device_not_healthy"
    raw=(witness.available_vram_bytes,witness.owner_resident_bytes,
         witness.reserved_for_service_bytes,witness.speculative_bytes_inflight,
         witness.queue_depth,witness.max_queue_depth)
    if (any(type(x) is not int or x<0 for x in raw) or
        witness.max_queue_depth<1 or
        witness.queue_depth>=witness.max_queue_depth or
        type(witness.memory_psi_full_avg10) not in (int,float) or
        not math.isfinite(witness.memory_psi_full_avg10) or
        witness.memory_psi_full_avg10>0):
        return [],"resource_pressure_or_telemetry_invalid"
    # available_vram_bytes is FREE VRAM now, as measured by R9700 sysfs,
    # not total VRAM. Refuse to consume owner-resident experts or the
    # service's non-negotiable working set. No evictions in this policy.
    free_after_reserve=(witness.available_vram_bytes-
                       witness.reserved_for_service_bytes-
                       witness.speculative_bytes_inflight-
                       MIN_VRAM_MARGIN)
    if free_after_reserve<BYTES_PER_EXPERT:
        return [],"no_spare_vram_without_eviction"
    if not (confidence.verified_split and confidence.source_runs>=2 and
            confidence.validation_rows>=MIN_USEFUL_SAMPLES and
            type(confidence.validation_precision) in (float,int) and
            math.isfinite(confidence.validation_precision) and
            confidence.validation_precision>=minimum_confidence and
            0<minimum_confidence<=1):
        return [],"unverified_or_low_precision_prediction"
    # Snapshot contains the known app-level residency bitmaps, never
    # current route labels. The shadow plan filters GPU residents first.
    if (snapshot.get("v")!=2 or snapshot.get("n_experts")!=N_EXPERTS):
        return [],"missing_full_residency_inventory"
    allowed=[]
    for expert in rank:
        prob=confidence.predicted_probability.get(expert)
        if (type(prob) not in (float,int) or not math.isfinite(prob) or
            prob<minimum_confidence or prob>1):
            continue
        allowed.append(expert)
        if len(allowed)>=2:break
    if not allowed:
        return [],"no_confident_candidates"
    gated=Limits(session=limit.session,pin=limit.pin,expected_pin=limit.expected_pin,
                 generation=limit.generation,expected_generation=limit.expected_generation,
                 enabled=True,max_candidates=min(limit.max_candidates,len(allowed),2),
                 nvme_budget=min(limit.nvme_budget,MAX_PER_EVENT_BYTES),
                 gpu_budget=min(limit.gpu_budget,MAX_PER_EVENT_BYTES,free_after_reserve),
                 gpu_headroom=min(limit.gpu_headroom,MAX_PER_EVENT_BYTES,free_after_reserve),
                 deadline_ms=limit.deadline_ms,nvme_GBps=limit.nvme_GBps,
                 upload_GBps=limit.upload_GBps,overhead_ms=limit.overhead_ms)
    nominations,status=plan(allowed,snapshot,gated)
    if not nominations:return [],"no_safe_candidate_"+status
    return nominations,"shadow_proposed_no_transfer"

def main():
    print("SHADOW_ONLY: import candidate() with independently attested witnesses; no hardware or production actions")
if __name__=="__main__":main()
