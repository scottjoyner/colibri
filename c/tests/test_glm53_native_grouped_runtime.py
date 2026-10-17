#!/usr/bin/env python3
"""GLM native six-tensor engine regression: resident / legacy / grouped / mmap.

Run with an already built GLM53 engine and an independent tiny streaming pair.
Never generates a fixture, rewrites a model or modifies running services.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--i4", type=Path, required=True)
    ap.add_argument("--deq", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--vulkan", action="store_true",
                    help="also activate the isolated R9700 Vulkan expert tier")
    args=ap.parse_args()
    ref=json.loads((args.i4/"ref.json").read_text())
    patch=args.i4/"patches.f32"
    if not patch.is_file():
        print("NO_GO: synthetic fixture patch missing",file=sys.stderr)
        return 2
    common=["--ids",",".join(map(str,ref["prompt"])),
            "--patches",str(patch),"--grid",
            "x".join(map(str,ref.get("grid",[4,4]))),"--greedy","3"]
    modes=[("resident",args.deq,"0",None),
           ("legacy",args.i4,"0",None),
           ("grouped",args.i4,"0","1"),
           ("vector",args.i4,"0","vector"),
           ("mmap_legacy",args.i4,"1",None),
           ("mmap_grouped",args.i4,"1","1"),
           ("mmap_vector",args.i4,"1","vector"),
           ("force",args.i4,"1","force"),
           ("force_vector",args.i4,"1","force-vector")]
    outputs={}
    for label,model,mapping,grouped in modes:
        environment={**os.environ,"GLM53_BITS":"32",
          "GLM53_EXPERT_GB":"0.000001","OMP_NUM_THREADS":"2",
          "COLI_MAP_EXPERTS":mapping,"USAGE_SAVE":"0",
          "COLI_USAGE":str(args.binary.parent/"glm53-native-tiny-usage-no-save")}
        environment.pop("GLM53_COALESCED_EXPERT_IO",None)
        if grouped: environment["GLM53_COALESCED_EXPERT_IO"]=grouped
        if args.vulkan:
            environment.update(COLI_VULKAN="1",COLI_VK_DEV="1",
                COLI_VK_CHAIN="0",COLI_VK_DENSE="0",
                COLI_VK_TIER_GB="0.02",COLI_VK_TIER_WARM="0")
        started=time.perf_counter()
        result=subprocess.run(
            [str(args.binary),"--model",str(model),*common],
            capture_output=True,text=True,env=environment,timeout=40)
        if result.returncode:
            print("NO_GO",label,"code",result.returncode,
                  "stderr tail",result.stderr[-400:],file=sys.stderr)
            return 2
        lines=[line.split() for line in result.stdout.splitlines() if line.strip()]
        fields={row[0]:row[1:] for row in lines}
        for k in ("teacher_forcing","greedy"):
            if k not in fields:
                print("NO_GO",label,"missing",k,file=sys.stderr)
                return 2
        outputs[label]={"teacher_forcing":fields["teacher_forcing"],
          "greedy":fields["greedy"],"grouped":fields.get("grouped",[]),
          "experts":fields.get("experts",[]),
          "elapsed_ms":round(1000*(time.perf_counter()-started),2)}
        print("MODE",label,"greedy",fields["greedy"],
              "grouped",fields.get("grouped",["off"]),flush=True)
    baseline=outputs["legacy"]
    for label,got in outputs.items():
        for field in ("teacher_forcing","greedy"):
            if got[field]!=baseline[field]:
                print("NO_GO",label,field,"differs from legacy",file=sys.stderr)
                return 2
    for label in ("grouped","vector","force","force_vector"):
        if not outputs[label]["grouped"] or int(outputs[label]["grouped"][1])<1:
            print("NO_GO",label,"no native grouped physical reads",file=sys.stderr)
            return 2
    # The telemetry line is printed whenever the flag is set, even if
    # the zero-copy mapping legitimately satisfies every expert load.
    if any(outputs[label]["grouped"] and
           int(outputs[label]["grouped"][1])!=0
           for label in ("mmap_grouped","mmap_vector")):
        print("NO_GO regular mmap bypassed by non-force grouped mode",file=sys.stderr)
        return 2
    print("PASS native eight-mode strict token parity and exact mmap bypass policy")
    if args.out:
        args.out.write_text(json.dumps({"schema":1,"models_read_only":True,
            "all_modes_token_exact":True,"vulkan":args.vulkan,
            "results":outputs},indent=2)+"\n")
    return 0

if __name__=="__main__":
    try: sys.exit(main())
    except Exception as error:
        print("NO_GO",type(error).__name__,str(error),file=sys.stderr)
        sys.exit(2)
