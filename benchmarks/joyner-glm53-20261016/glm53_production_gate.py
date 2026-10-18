#!/usr/bin/env python3
"""Read-only GLM 5.3 24/7 admission evidence. NEVER starts/adopts a service.

JSON soak reports are untrusted assertions; passing the automatic checks
can only request an independent operator review, NEVER auto-deploy.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

EXCLUSIVE=re.compile(r"exclusive4(control|fair)-([0-9a-f]{12})\.json\Z")
SHA256=re.compile(r"[0-9a-f]{64}\Z")
EXPECTED="16 11 21 13 17 11 21 13"
MIN_PAIRS=3
MIN_SPEED_TOK_S=3.0
MIN_SOAK_HOURS=72.0
MIN_REQUESTS=1000
MAX_P95_MS_PER_TOKEN=750.0
MAX_RECOVERY_SEC=60.0

def load_json(path:Path)->dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>256*1024:
        raise ValueError("unsafe or oversized evidence file")
    item=json.loads(path.read_text())
    if not isinstance(item,dict):
        raise ValueError("evidence is not a JSON object")
    return item

def receipt(path:Path)->dict:
    m=EXCLUSIVE.fullmatch(path.name)
    if not m:raise ValueError("not an exclusive benchmark receipt")
    d=load_json(path)
    if d.get("mode")!=path.stem or d.get("fair_layers")!=int(m[1]=="fair"):
        raise ValueError("receipt identity or policy mismatch")
    if (not d.get("completed") or d.get("isolated") is not True or
        d.get("competitors_detected")!=[] or d.get("greedy_ids")!=EXPECTED or
        d.get("gpu_routed")!=3696 or d.get("exit_code")!=0):
        raise ValueError("incomplete, contaminated or output-mismatched receipt")
    digest=d.get("binary_sha256")
    if not isinstance(digest,str) or not SHA256.fullmatch(digest):
        raise ValueError("missing pinned binary SHA256")
    speed=d.get("tok_s")
    if type(speed) not in (float,int) or not 0<float(speed)<=100:
        raise ValueError("invalid throughput")
    if type(d.get("min_available_kib")) is not int or d["min_available_kib"]<900000:
        raise ValueError("host RAM floor violated")
    if type(d.get("max_psi_full_avg10")) not in (int,float) or d["max_psi_full_avg10"]>=5:
        raise ValueError("memory pressure watchdog violation")
    return {"case":m[1],"round":m[2],"speed":float(speed),
            "binary_sha256":digest,"greedy":d["greedy_ids"]}

def evaluate_benchmarks(folder:Path)->tuple[dict,list[str]]:
    reasons=[]
    paired=[]
    candidates={}
    rejected=[]
    if folder.is_symlink() or not folder.is_dir():
        return {"paired_runs":0},["no safe benchmark evidence directory"]
    for path in sorted(folder.glob("exclusive4*.json")):
        if not EXCLUSIVE.fullmatch(path.name):continue
        try:
            r=receipt(path)
            candidates.setdefault(r["round"],{})[r["case"]]=r
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
            rejected.append({"filename":path.name,"reason":str(exc)[:140]})
    for round_id,cases in candidates.items():
        if set(cases)!={"control","fair"}:
            rejected.append({"round":round_id,"reason":"unpaired experiment"})
            continue
        ctl,fair=cases["control"],cases["fair"]
        if ctl["binary_sha256"]!=fair["binary_sha256"]:
            rejected.append({"round":round_id,"reason":"binary changed within pair"})
            continue
        paired.append({"round":round_id,"control_tok_s":ctl["speed"],
                       "fair_tok_s":fair["speed"],"binary_sha256":fair["binary_sha256"]})
    if len(paired)<MIN_PAIRS:reasons.append("at least 3 uncontaminated paired physical benchmarks required")
    if len({r["binary_sha256"] for r in paired})>1:
        reasons.append("benchmark rounds target different binary builds; separate each release candidate")
    speeds=[r["fair_tok_s"] for r in paired]
    if not speeds or min(speeds)<MIN_SPEED_TOK_S:
        reasons.append("minimum fair-share throughput below 3 tokens/s")
    # All legacy 8-token receipts use one synthetic prompt. New diverse prompt
    # ID and reference-logit tests must be collected separately for admission.
    reasons.append("no independently verified diverse natural-language prompt and reference-logit parity evidence")
    return {"paired_runs":len(paired),"fair_min_tok_s":min(speeds) if speeds else None,
            "fair_median_tok_s":statistics.median(speeds) if speeds else None,
            "pairs":paired,"rejected":rejected},reasons

def evaluate_soak(path:Path|None,now:datetime)->tuple[dict,list[str]]:
    if path is None:return {"provided":False},["no 72-hour serving soak receipt"]
    try:
        d=load_json(path)
        if d.get("schema")!=1:raise ValueError("unsupported soak schema")
        start=datetime.fromisoformat(d["start_utc"].replace("Z","+00:00"))
        end=datetime.fromisoformat(d["end_utc"].replace("Z","+00:00"))
        if start.tzinfo is None or end.tzinfo is None or not start<end<=now:
            raise ValueError("invalid UTC soak time interval")
        hours=(end-start).total_seconds()/3600
        requests=d["requests"]
        failures=d["failures"]
        prompts=d["natural_language_requests"]
        p95=d["p95_ms_per_token"]
        recover=d["max_restart_recovery_seconds"]
        auth_denials=d["unauthorized_denials_verified"]
        tenant_violations=d["tenant_isolation_failures"]
        crash=d["unrecovered_crashes"]
        fields=[requests,failures,prompts,auth_denials,tenant_violations,crash]
        if any(type(x) is not int or x<0 for x in fields):
            raise ValueError("invalid soak counts")
        if failures>requests or prompts>requests or requests==0:
            raise ValueError("inconsistent request totals")
        if (type(p95) not in (int,float) or type(recover) not in (int,float)
            or not 0<=p95<1e8 or not 0<=recover<1e8):
            raise ValueError("invalid latency / recovery values")
    except (OSError,KeyError,ValueError,TypeError,json.JSONDecodeError) as exc:
        return {"provided":True,"valid":False},["soak evidence invalid: "+str(exc)[:160]]
    reasons=[]
    if hours<MIN_SOAK_HOURS:reasons.append("soak duration below 72 hours")
    if requests<MIN_REQUESTS or prompts<100:reasons.append("insufficient load and natural-language coverage")
    if failures/max(requests,1)>.001:reasons.append("request failure rate above 0.1%")
    if p95>MAX_P95_MS_PER_TOKEN:reasons.append("p95 token latency exceeds 750 ms")
    if recover>MAX_RECOVERY_SEC:reasons.append("restart recovery exceeds 60 seconds")
    if auth_denials<10:reasons.append("negative authentication/access tests absent")
    if tenant_violations or crash:reasons.append("tenant isolation failures or unrecovered crashes")
    return {"provided":True,"valid":True,"hours":round(hours,2),
            "requests":requests,"failed_requests":failures,
            "p95_ms_per_token":p95,"unrecovered_crashes":crash},reasons

def report(receipts:Path,soak:Path|None=None,*,now:datetime|None=None)->dict:
    if now is None:now=datetime.now(timezone.utc)
    b,br=evaluate_benchmarks(receipts)
    s,sr=evaluate_soak(soak,now)
    reasons=br+sr
    # The admission authority is external; this tool never issues 'GO'.
    reasons.append("independent signed source/checkpoint attestation and multi-user failover/rollback acceptance outstanding")
    return {"schema":1,"engine":"glm53","role":"read_only_evidence_gate",
            "benchmark":b,"soak":s,"decision":"HOLD",
            "evidence_attested":False,"automatic_admission_supported":False,
            "blockers":reasons,"can_start_server":False,
            "can_admit_fleet":False,"can_deploy":False,
            "next_action":"independent operator review only after every gate has external verifiable evidence"}

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipts",type=Path,required=True)
    ap.add_argument("--soak",type=Path)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    outcome=report(a.receipts,a.soak)
    payload=json.dumps(outcome,indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.is_symlink() or a.output.exists():
            raise SystemExit("NO_GO cannot overwrite evidence output")
        with a.output.open("x") as fd:fd.write(payload)
    else:print(payload,end="")
    return 0  # successful inspection, NEVER fleet admission
if __name__=="__main__":raise SystemExit(main())
