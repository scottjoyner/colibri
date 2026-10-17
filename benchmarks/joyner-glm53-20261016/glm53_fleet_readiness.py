#!/usr/bin/env python3
"""GLM-5.3 read-only fleet admission preflight; NEVER dispatches inference.

This is a measurement and HOLD receipt, not a remote start, security
authorization, load test, cost estimate, or production rollout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
import sys
from typing import Any

EXPECTED_SHARDS = 62
HEADER_LIMIT = 64 * 1024 * 1024
GIB = 1024 ** 3
MODEL_PIN = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
MODEL_SHARD_RE = re.compile(r"model-\d{5}-of-00062\.safetensors\Z")
MODEL_EXPERT = ".mlp.experts."

def read_meminfo(root: Path) -> dict[str,int]:
    values: dict[str,int] = {}
    for line in (root / "proc/meminfo").read_text().splitlines():
        match=re.fullmatch(r"([A-Za-z_()]+):\s+(\d+)\s+kB",line)
        if match: values[match[1]] = int(match[2])*1024
    for key in ("MemTotal","MemAvailable","SwapTotal","SwapFree"):
        if key not in values:raise ValueError(f"missing {key}")
    return values

def read_pressure(root:Path) -> dict[str,dict[str,float]]:
    result={}
    for row in (root/"proc/pressure/memory").read_text().splitlines():
        values=dict((key,float(value)) for key,value in
                    (part.split("=",1) for part in row.split()[1:]))
        result[row.split()[0]]={k:values[k] for k in ("avg10","avg60","avg300")}
    if set(result)!={"some","full"}:
        raise ValueError("missing memory pressure categories")
    return result

def physical_gpu(root: Path) -> dict[str, Any]:
    """Only report actual VRAM sysfs values, never assume Vulkan name/ordinal."""
    devices=[]
    base=root/"sys/class/drm"
    for card in sorted(base.glob("card[0-9]*")):
        device=card/"device"
        used=device/"mem_info_vram_used"
        total=device/"mem_info_vram_total"
        if not (used.is_file() and total.is_file()):continue
        raw_total=int(total.read_text().strip())
        raw_used=int(used.read_text().strip())
        if raw_total<=0 or raw_used<0 or raw_used>raw_total:
            raise ValueError("invalid VRAM readings")
        product=(device/"device").read_text().strip() if (device/"device").exists() else None
        vendor=(device/"vendor").read_text().strip() if (device/"vendor").exists() else None
        devices.append(dict(name=card.name,device=product,vendor=vendor,
                            total_bytes=raw_total,used_bytes=raw_used,
                            available_bytes=raw_total-raw_used))
    candidates=[g for g in devices if g["vendor"]=="0x1002" and
                g["device"]=="0x7551" and g["total_bytes"]>=24*GIB]
    if len(candidates)!=1:
        return {"selected":None,"candidates":len(candidates),
                "device_count":len(devices),"reason":"R9700 hardware identity not uniquely verified"}
    return {"selected":candidates[0],"candidates":1,
            "device_count":len(devices),"reason":None}

def inspect_shards(model: Path) -> dict[str, Any]:
    if not model.is_dir() or model.is_symlink():
        raise ValueError("source model directory missing or symlinked")
    shards=sorted(model.glob("model-*-of-00062.safetensors"))
    if len(shards)!=EXPECTED_SHARDS or not all(MODEL_SHARD_RE.fullmatch(f.name) for f in shards):
        raise ValueError(f"NO_GO expected exactly {EXPECTED_SHARDS} GLM source shards")
    exact=[f"model-{i:05d}-of-00062.safetensors" for i in range(1,63)]
    if [p.name for p in shards]!=exact:
        raise ValueError("shard identity incomplete/duplicate")
    categories={k:{"bytes":0,"tensors":0} for k in
                ("routed_experts","dense_or_other","output","vision")}
    header_digest=hashlib.sha256()
    file_total=0
    expert_tensors=0
    for shard in shards:
        fd=os.open(shard,os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            st=os.fstat(fd)
            if (not stat.S_ISREG(st.st_mode) or st.st_size<8 or
                st.st_size>500*GIB):
                raise ValueError(f"invalid source file: {shard.name}")
            file_total+=st.st_size
            header_len=struct.unpack("<Q",os.pread(fd,8,0))[0]
            if not 0<header_len<=HEADER_LIMIT or header_len>st.st_size-8:
                raise ValueError(f"invalid header: {shard.name}")
            header=os.pread(fd,header_len,8)
            if len(header)!=header_len:
                raise ValueError(f"short header: {shard.name}")
            header_digest.update(shard.name.encode()+b"\0")
            header_digest.update(struct.pack("<Q",header_len)+header)
            fields=json.loads(header)
            intervals=[]
            for key,meta in fields.items():
                if key=="__metadata__":continue
                span=meta.get("data_offsets")
                if not isinstance(span,list) or len(span)!=2:raise ValueError("invalid tensor offsets")
                lo,hi=span
                if (not isinstance(lo,int) or not isinstance(hi,int) or
                    lo<0 or hi<lo or hi>st.st_size-8-header_len):
                    raise ValueError("tensor outside original safetensor shard")
                tag=("routed_experts" if MODEL_EXPERT in key else
                     "vision" if ".visual." in key else
                     "output" if ".lm_head." in key or ".embed_tokens." in key else
                     "dense_or_other")
                if hi==lo: raise ValueError("empty tensor extent")
                intervals.append((lo,hi))
                categories[tag]["bytes"]+=hi-lo
                categories[tag]["tensors"]+=1
                if tag=="routed_experts":expert_tensors+=1
            intervals.sort()
            if any(intervals[j][0]<intervals[j-1][1] for j in range(1,len(intervals))):
                raise ValueError("overlapping tensor spans in safetensor shard")
            st_after=os.fstat(fd)
            if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(
                 st_after.st_dev,st_after.st_ino,st_after.st_size,st_after.st_mtime_ns):
                raise ValueError("source descriptor changed during header audit")
        finally:
            os.close(fd)
    if expert_tensors!=12384*6:
        raise ValueError(f"expected 74304 expert tensors, found {expert_tensors}")
    return dict(expected_source_pin=MODEL_PIN,source="local_headers_only",
                source_pin_cryptographically_verified=False,
                shards=EXPECTED_SHARDS,total_shard_file_bytes=file_total,
                header_sha256=header_digest.hexdigest(),categories=categories)

def readiness(model:Path,root:Path,*,host_reserve_gib:float=4,
              gpu_reserve_gib:float=4,expert_cache_gib:float=.10,
              gpu_tier_gib:float=2,staging_gib:float=.5) -> dict[str,Any]:
    budgets=(host_reserve_gib,gpu_reserve_gib,expert_cache_gib,gpu_tier_gib,staging_gib)
    if not all(math.isfinite(value) and 0<=value<=128 for value in budgets):
        raise ValueError("non-finite, negative or unbounded budget request")
    layout=inspect_shards(model)
    mem=read_meminfo(root)
    psi=read_pressure(root)
    gpu=physical_gpu(root)
    reasons=[]
    if mem["MemAvailable"]<int((host_reserve_gib+expert_cache_gib+staging_gib)*GIB):
        reasons.append("insufficient host MemAvailable for explicit reserve + cache + staging")
    if mem["SwapTotal"]>mem["SwapFree"]:
        reasons.append("host swap already in use; reject new high-memory placement")
    if max(psi["some"]["avg10"],psi["full"]["avg10"])>0:
        reasons.append("nonzero current memory pressure")
    if not gpu["selected"]:
        reasons.append("R9700 GPU identity or exact VRAM availability not verified")
    elif gpu["selected"]["available_bytes"]<int((gpu_reserve_gib+gpu_tier_gib)*GIB):
        reasons.append("insufficient R9700 VRAM for reserve + expert tier")
    # Explicit release blocker irrespective of static capacity. 32GiB VRAM
    # is not a proof of the dense chain, checkpoint, KV and expert pool
    # fitting together, nor proof of a safe full model launch.
    reasons.extend([
       "NO-GO: expected upstream commit is not verified by a complete source SHA manifest",
       "NO-GO: prior bounded 8-token synthetic full-GLM run (0.630 tok/s) does not establish sustained natural-language production readiness",
       "NO-GO: no full-model reference logit parity, sustained natural-dialogue tok/s or p99 layer deadline acceptance",
       "NO-GO: no authenticated fleet admission, fencing or rollback verification",
    ])
    return {
      "schema":1,"mode":"read_only_fleet_runtime_preflight","host":socket.gethostname(),
      "checkpoint":layout,"memory":{
        "total_bytes":mem["MemTotal"],"available_bytes":mem["MemAvailable"],
        "swap_used_bytes":mem["SwapTotal"]-mem["SwapFree"],
        "pressure_avg10":{k:v["avg10"] for k,v in psi.items()}},
      "gpu":gpu,"budget_request_gib":{
        "host_reserve":host_reserve_gib,"gpu_reserve":gpu_reserve_gib,
        "expert_cache":expert_cache_gib,"gpu_tier":gpu_tier_gib,"staging":staging_gib},
      "production_model_loaded":None,"fleet_admitted":None,
      "inference_started_by_preflight":False,"fleet_admission_modified":False,
      "writes_to_model":False,"starts_server":False,"decision":"HOLD",
      "blockers":reasons}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--sys-root",type=Path,default=Path("/"))
    ap.add_argument("--host-reserve-gib",type=float,default=4)
    ap.add_argument("--gpu-reserve-gib",type=float,default=4)
    ap.add_argument("--expert-cache-gib",type=float,default=.10)
    ap.add_argument("--gpu-tier-gib",type=float,default=2)
    ap.add_argument("--staging-gib",type=float,default=.5)
    a=ap.parse_args()
    try:
        report=readiness(a.model,a.sys_root,
             host_reserve_gib=a.host_reserve_gib,
             gpu_reserve_gib=a.gpu_reserve_gib,
             expert_cache_gib=a.expert_cache_gib,
             gpu_tier_gib=a.gpu_tier_gib,staging_gib=a.staging_gib)
        print(json.dumps(report,indent=2,sort_keys=True))
        return 0 # successful *audit*, never means deploy allowed
    except Exception as e:
        print(json.dumps({"schema":1,"mode":"read_only_fleet_runtime_preflight",
             "decision":"NO_GO","blockers":[type(e).__name__+": "+str(e)]},sort_keys=True))
        return 2

if __name__=="__main__":
    sys.exit(main())
