#!/usr/bin/env python3
"""Real GLM5.3 trace-predicted 4MiB O_DIRECT expert-read replay, no live prefetch.

Prediction -> freeze nominations -> optional O_DIRECT -> score against native
label. Neither a router hook nor a full-expert upload benchmark.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, math, os, pathlib
import re, statistics, struct, sys, time
from dataclasses import asdict
from typing import Any

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from replay_expert_prefetch import read_trace, compile_stats, N_EXPERTS
from select_expert_policy import scores
from shadow_tier_plan import plan, Limits
from calibrated_prefetch_budget import fit, Calibration
from bounded_shadow_reader import BoundedShadowReader

PIN="46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
MODEL=pathlib.Path("/srv/joyner-gen5/models/glm53_colibri_i4")
PRIVATE=pathlib.Path("/home/joyner/provider-r9700/colibri-fork-expert-trace-20261016/benchmarks/joyner-glm53-20261016")
OUTPUT=pathlib.Path("/home/joyner/provider-r9700/colibri-pool-fixture-evidence-20261017")
PATTERN=re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$")
WINDOW=4*1024*1024
BLOCK=4096
FROZEN=(1.0,2.0,0.25)

def percentile(vals:list[float],frac:float)->float|None:
    if not vals:return None
    ordered=sorted(vals)
    return round(ordered[max(0,math.ceil(len(ordered)*frac)-1)],3)

def ms_psi()->float:
    for line in pathlib.Path("/proc/pressure/memory").read_text().splitlines():
        if line.startswith("full "):
            return float(dict(part.split("=") for part in line.split()[1:])["avg10"])
    return float("inf")

def safety():
    available=0
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available=int(line.split()[1]);break
    if available<1024*1024 or ms_psi()>2.0:
        raise RuntimeError("NO_GO: host memory pressure")

def validate_model()->dict[tuple[int,int],tuple[pathlib.Path,int]]:
    source=json.loads((MODEL/".source-revision.json").read_text())
    if source!={"repo":"zai-org/GLM-5.3-Flash","revision":PIN}:
        raise RuntimeError("NO_GO: unexpected checkpoint revision")
    done=json.loads((MODEL/".converted.json").read_text())
    digests=json.loads((MODEL/".converted-sha256.json").read_text())
    if len(done)!=62 or set(done)!=set(digests):
        raise RuntimeError("NO_GO: incomplete shard inventory")
    mount=os.popen("findmnt -n -T /srv/joyner-gen5 -o SOURCE").read().strip()
    if mount!="/dev/nvme0n1":
        raise RuntimeError("NO_GO: checkpoint not on pinned Gen5 SSD")
    experts={}
    for name in sorted(done):
        if not name.startswith("model-") or not name.endswith(".safetensors"):
            raise RuntimeError("NO_GO: unexpected shard")
        file=MODEL/name
        with file.open("rb") as stream:
            prefix=stream.read(8)
            if len(prefix)!=8:raise RuntimeError("bad header length")
            n=struct.unpack("<Q",prefix)[0]
            if n>50_000_000:raise RuntimeError("header too big")
            header=json.loads(stream.read(n))
        size=file.stat().st_size
        for key,blob in header.items():
            match=PATTERN.match(key)
            if not match or blob.get("dtype")!="U8":continue
            l,e=map(int,match.groups())
            first,last=blob["data_offsets"]
            start=8+n+first
            aligned=start//BLOCK*BLOCK
            if aligned+WINDOW>size or last-first<WINDOW:
                raise RuntimeError("NO_GO invalid expert aligned 4MiB region")
            if (l,e) in experts:
                raise RuntimeError("NO_GO duplicate expert offset")
            experts[(l,e)]=(file,aligned)
    if not experts:raise RuntimeError("NO_GO no real expert index")
    return experts

def build_policy(policy,event,history,freq,cooc)->list[int]:
    if policy=="fusion":
        return scores(event,history,freq,cooc,FROZEN)
    layer=event["layer"]
    rank=[]
    if policy=="prior_token":
        prev=history.get(("token",layer,event["row"]))
        if prev:rank.extend(prev["ids"])
    elif policy=="cross_layer":
        prev=history.get(("call",event["call"]-1,event["row"]))
        if prev and prev["layer"]==layer-1:
            weights={}
            for source in prev["ids"]:
                for eid,n in cooc.get((layer,source),{}).items():
                    weights[eid]=weights.get(eid,0)+n
            rank.extend(sorted(weights,key=lambda eid:(-weights[eid],eid)))
    else:raise ValueError("unknown policy")
    rank.extend(sorted(freq[layer],key=lambda eid:(-freq[layer][eid],eid)))
    rank.extend(range(N_EXPERTS))
    return list(dict.fromkeys(rank))

def read_one(fd:int,offset:int)->float:
    import mmap
    if offset<0 or offset%BLOCK:
        raise ValueError("NO_GO O_DIRECT expert offset not aligned")
    # mmap anonymous mapping is page-aligned; preadv fills it without caching.
    buf=mmap.mmap(-1,WINDOW,access=mmap.ACCESS_WRITE)
    try:
        start=time.perf_counter()
        count=os.preadv(fd,[buf],offset)
        elapsed=(time.perf_counter()-start)*1000
        if count!=WINDOW:
            raise OSError(f"short O_DIRECT expert read {count}/{WINDOW}")
        return elapsed
    finally:buf.close()

def benchmark(mode:str,events:list[dict],freq,cooc,index,window_ms:float,read:bool,calibration:Calibration|None=None)->dict:
    limits=Limits(session="single-run-shadow-only",
                  pin=PIN,expected_pin=PIN,generation=1,expected_generation=1,
                  enabled=True,max_candidates=2,nvme_budget=2*14155776,
                  gpu_budget=2*14155776,gpu_headroom=2*14155776,
                  deadline_ms=window_ms,nvme_GBps=11.319,
                  upload_GBps=8.0,overhead_ms=0.5)
    history={};nominations=[];utility=0;correct_cold=0;cold_total=0
    actual_reads=0;elapsed=[];step_latency=[];gated_second=0
    def disk_sector_bytes():
        return int(pathlib.Path("/sys/block/nvme0n1/stat").read_text().split()[2])*512
    # The prepared frozen plan only consults history and cached bitmaps
    # before it reads or compares the current ground-truth expert ids.
    for ev in events:
        rank=build_policy(mode,ev,history,freq,cooc)
        picked,reason=plan(rank,ev,limits)
        assert reason in ("proposed","outside_envelope")
        # Calibration was fitted on run05, NEVER on the held-out run06.
        # Gate the second proposed candidate *before* consulting ground truth.
        if calibration is not None and len(picked)>1 and not calibration.permitted(ev["layer"],1):
            picked=picked[:1]
            gated_second+=1
        targets={x["expert"] for x in picked}
        if any((ev["layer"],eid) not in index for eid in targets):
            raise RuntimeError("NO_GO mapped real expert missing")
        nominations.append((ev,picked,reason))
        prior_targets=set(ev["ids"])
        utility+=len(prior_targets & targets)
        cold={eid for eid,tier in zip(ev["ids"],ev["resident"]) if tier==0}
        cold_total+=len(cold)
        correct_cold+=len(cold & targets)
        history[("token",ev["layer"],ev["row"])]=ev
        history[("call",ev["call"],ev["row"])]=ev
    # Never score a candidate before all selections have been frozen.
    if read:
        safety()
        disk_before=disk_sector_bytes()
        handles={}
        try:
            needed={index[(ev["layer"],p["expert"])][0]
                    for ev,picks,_ in nominations for p in picks
                    if p["tier"]=="unknown_disk_or_page_cache"}
            handles={path:os.open(path,os.O_RDONLY|os.O_DIRECT|os.O_CLOEXEC)
                     for path in needed}
            with BoundedShadowReader(
                    session="single-run-shadow-only",pin=PIN,generation=1,
                    read_fn=read_one,max_workers=2) as executor:
                for row,(ev,picks,_) in enumerate(nominations):
                    requests=[]
                    begin=time.perf_counter()
                    for p in picks:
                        if p["tier"]!="unknown_disk_or_page_cache":continue
                        f,off=index[(ev["layer"],p["expert"])]
                        requests.append((handles[f],off))
                    finished=executor.read_batch(
                        session="single-run-shadow-only",pin=PIN,
                        generation=1,requests=requests)
                    elapsed.extend(finished)
                    actual_reads+=len(finished)
                    if finished:step_latency.append((time.perf_counter()-begin)*1000)
                    if row%16==0:safety()
                if actual_reads!=executor.completed:
                    raise RuntimeError("NO_GO incomplete admitted disk reads")
        finally:
            for fd in handles.values():os.close(fd)
        physical_sector_bytes=disk_sector_bytes()-disk_before
    else:
        physical_sector_bytes=None
    submitted=sum(len(picks) for _,picks,_ in nominations)
    planned_disk=sum(p["tier"]=="unknown_disk_or_page_cache"
                     for _,picks,_ in nominations for p in picks)
    return {"policy":mode,"rows":len(events),"candidates":submitted,
            "adaptive_second_gated_events":gated_second,
            "candidate_precision":round(utility/max(1,submitted),5),
            "selected_recall":round(utility/(8*len(events)),5),
            "cold_selected_upper_coverage":round(correct_cold/max(1,cold_total),5),
            "predicted_unknown_tier_reads":planned_disk,
            "physical_O_DIRECT_reads":actual_reads,
            "physical_O_DIRECT_GB":round(actual_reads*WINDOW/1e9,4),
            "block_device_sector_read_delta_GB":round(physical_sector_bytes/1e9,4) if physical_sector_bytes is not None else None,
            "physical_read_ms_p50":percentile(elapsed,.5),
            "physical_read_ms_p99":percentile(elapsed,.99),
            "two_reader_step_ms_p50":percentile(step_latency,.5),
            "two_reader_step_ms_p99":percentile(step_latency,.99)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rows",type=int,default=96)
    ap.add_argument("--window-ms",type=float,default=10.)
    ap.add_argument("--execute-io",action="store_true")
    ap.add_argument("--policy-order",default="prior_token,cross_layer,fusion")
    ap.add_argument("--adaptive-fusion",action="store_true",
                    help="fit second-candidate utility on independent run05; gate fusion ONLY")
    ap.add_argument("--second-min-prob",type=float,default=0.40)
    ap.add_argument("--adaptive-only",action="store_true",
                    help="measure frozen fusion baseline plus validation-gated fusion only")
    ap.add_argument("--adaptive-first",action="store_true",
                    help="reverse measurement order of two fusion variants")
    ap.add_argument("--out",type=pathlib.Path)
    args=ap.parse_args()
    if args.rows<16 or args.rows>512 or args.window_ms<=0:
        ap.error("rows must 16..512 and deadline positive")
    if args.adaptive_only and not args.adaptive_fusion:
        ap.error("adaptive-only requires adaptive-fusion")
    safety()
    index=validate_model()
    names=sorted(PRIVATE.glob("glm53_native_expert_trace_20261016_run*.jsonl"))
    if len(names)!=6:raise RuntimeError("NO_GO missing independent runs")
    train=[read_trace(p) for p in names[:4]]
    heldout=names[5]
    if "run06_v2" not in heldout.name:raise RuntimeError("NO_GO unexpected holdout")
    events=read_trace(heldout)[:args.rows]
    if any(ev["v"]!=2 for ev in events):raise RuntimeError("NO_GO no full v2 cache inventory")
    freq,cooc=compile_stats(train)
    calibration=None
    if args.adaptive_fusion:
        validation=names[4]
        if "run05" not in validation.name:
            raise RuntimeError("NO_GO unexpected validation run")
        validation_events=read_trace(validation)
        calibration=fit(validation_events,
                        lambda ev,h:build_policy("fusion",ev,h,freq,cooc),
                        threshold=args.second_min_prob)
    policies=tuple(args.policy_order.split(","))
    if len(policies)!=3 or set(policies)!={"prior_token","cross_layer","fusion"}:
        ap.error("policy-order must be a permutation of prior_token,cross_layer,fusion")
    # The unmodified fusion run remains an in-output baseline, and the
    # independently calibrated fusion run is added only when explicitly asked.
    if args.adaptive_only:
        modes=["fusion_validation_gated","fusion"] if args.adaptive_first else [
            "fusion","fusion_validation_gated"]
        results=[]
        for mode in modes:
            result=benchmark("fusion",events,freq,cooc,index,args.window_ms,
                             args.execute_io,
                             calibration if mode=="fusion_validation_gated" else None)
            result["policy"]=mode
            results.append(result)
    else:
        results=[benchmark(p,events,freq,cooc,index,args.window_ms,args.execute_io) for p in policies]
        if calibration is not None:
            calibrated=benchmark("fusion",events,freq,cooc,index,
                                 args.window_ms,args.execute_io,calibration)
            calibrated["policy"]="fusion_validation_gated"
            results.append(calibrated)
    report={"schema":1,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "source_pin":PIN,"verified_shards":62,"indexed_real_expert_up_tensors":len(index),
            "train_runs":[p.name for p in names[:4]],"heldout_run":heldout.name,
            "heldout_sha256":hashlib.sha256(heldout.read_bytes()).hexdigest(),
            "rows":len(events),"read_window_bytes":WINDOW,"max_read_workers":2,
            "deadline_ms_assumed":args.window_ms,"upload_GBps_assumed":8.0,
            "physical_io_enabled":args.execute_io,
            "adaptive_validation":({
                 "validation_run":names[4].name,
                 "validation_sha256":hashlib.sha256(names[4].read_bytes()).hexdigest(),
                 "threshold_frozen_before_heldout":calibration.threshold,
                 "global_second_success_probability":round(calibration.global_second_probability,5),
                 "calibration_events":calibration.train_rows,
                 "layer_probabilities":{str(k):round(v,5) for k,v in calibration.layer_second_probability.items()}
              } if calibration is not None else None),
            "comparison":results,
            "limitations":["Independent synthetic-ID GLM traces, not general workload",
                          "Only 4MiB up_proj tensor windows read, NOT complete 14MiB routed expert",
                          "Physical shadow read is not a live expert prefetch or GPU upload",
                          "Linux O_DIRECT bypasses Linux page cache but not SSD controller cache",
                          "Read requests processed after all nominees frozen, not overlapped with live inference",
                          "Candidate benefit judged ex post from native route; no runtime speedup proven",
                          "SSD read p99 cannot validate speculative layer deadlines without real compute overlap"],
            "production_changed":False}
    text=json.dumps(report,indent=2)+"\n"
    if args.out:args.out.write_text(text)
    print(text)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("NO_GO:",type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(2)
