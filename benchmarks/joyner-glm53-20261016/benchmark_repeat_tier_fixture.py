#!/usr/bin/env python3
"""Native GLM-5.3 repeated-route promotion parity under a bounded R9700 tier.

Requires a private *tiny* fixture. No full-model launch, no network or service
restart, no writes to checkpoint, never an admission decision.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

def execute(binary:Path,fixture:Path,repeat:bool,staged:bool):
    binary=binary.resolve(strict=True)
    fixture=fixture.resolve(strict=True)
    ref=json.loads((fixture/"ref.json").read_text())
    patch=fixture/"patches.f32"
    assert patch.is_file() and binary.is_file()
    env={**os.environ,"GLM53_BITS":"32","GLM53_EXPERT_GB":"0.000001",
         "GLM53_COALESCED_EXPERT_IO":"vector",
         "COLI_MAP_EXPERTS":"0","OMP_NUM_THREADS":"2",
         "COLI_VULKAN":"1","COLI_VK_DEV":"1","COLI_VK_CHAIN":"0",
         "COLI_VK_DENSE":"0","COLI_VK_TIER_GB":"0.02",
         "COLI_VK_TIER_WARM":"0","COLI_VK_TIER_RATE":"16",
         "COLI_VK_STAGED":"1" if staged else "0","COLI_VK_UP_IMPORT":"0",
         "COLI_VK_TIER_REPEAT_ADMIT":"1" if repeat else "0",
         "USAGE_SAVE":"0"}
    cmd=[str(binary),"--model",str(fixture),"--ids",
         ",".join(str(v) for v in ref["prompt"]),"--patches",str(patch),
         "--grid","x".join(map(str,ref["grid"])),"--greedy","3"]
    process=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=35)
    if process.returncode:
        raise ValueError(f"NO_GO native runtime failed: {process.returncode}, stderr tail: {process.stderr[-450:]}")
    out={row[0]:row[1:] for line in process.stdout.splitlines()
         if (row:=line.split())}
    for key in ("teacher_forcing","greedy","experts","grouped"):
        if key not in out:raise ValueError("NO_GO missing native telemetry "+key)
    if int(out["grouped"][1])<1:raise ValueError("NO_GO vector IO not exercised")
    tier=[line for line in process.stderr.splitlines() if "[VK] tier glm53 run:" in line]
    if not tier:raise ValueError("NO_GO Vulkan tier not observed")
    line=tier[-1]
    routes=re.search(r"device (\d+) of (\d+) routed experts",line)
    uploads=re.search(r"uploads (\d+) ",line)
    skips=re.search(r"experimental repeat-first: (\d+) one-off promotions skipped",line)
    if not routes or not uploads:raise ValueError("NO_GO missing GPU-routing telemetry")
    if repeat and skips is None:raise ValueError("NO_GO repeat gate telemetry absent")
    if not repeat and skips is not None:raise ValueError("NO_GO default path polluted by repeat gate")
    return {"mode":"repeat_first" if repeat else "baseline",
            "teacher_forcing":out["teacher_forcing"],"greedy":out["greedy"],
            "grouped_loads":int(out["grouped"][1]),
            "gpu_served":int(routes[1]),"routed":int(routes[2]),
            "gpu_uploads":int(uploads[1]),
            "repeat_skipped":int(skips[1]) if skips else 0}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--staged",action="store_true")
    a=p.parse_args()
    reference=execute(a.binary,a.fixture,False,a.staged)
    candidate=execute(a.binary,a.fixture,True,a.staged)
    for field in ("teacher_forcing","greedy"):
        if reference[field]!=candidate[field]:
            raise ValueError("NO_GO token parity failure on "+field)
    report={"schema":1,"source":"tiny native GLM engine real Vulkan execution",
      "staged_uploads":a.staged,
      "fixture_ref_sha256":hashlib.sha256((a.fixture/"ref.json").read_bytes()).hexdigest(),
      "baseline":reference,"repeat_first":candidate,
      "strict_token_parity":True,"full_model_speedup_proven":False,
      "production_services_changed":False}
    result=json.dumps(report,indent=2)+"\n"
    print(result)
    if a.out:
        if a.out.exists():raise ValueError("NO_GO refusing existing output")
        a.out.write_text(result)
if __name__=="__main__":
    main()
