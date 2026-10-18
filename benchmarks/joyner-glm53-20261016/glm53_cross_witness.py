#!/usr/bin/env python3
"""Out-of-band correlation of independently hosted Joyner availability monitors.

Two separate network observers provide *corroboration* of reachability, NOT
independent signing custody, loaded-model proof, routing admission, fencing,
failover authorization, or any power-cycle authority.

Read-only: local state read and a pinned StrictHostKeyChecking SSH cat to a
second user-controlled node; never writes to either node or dispatches GLM.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
import time

SCHEMA=1
MAX_STATE=16384
MAX_AGE_SECONDS=180
MAX_CLOCK_SKEW_SECONDS=90
OWNERS=frozenset(("x1-370","xwing"))
STATUS=frozenset(("UNREACHABLE","HEALTH_FAILED","HOST_HTTP_ONLY",
                  "MODEL_NOT_LISTED","MODEL_CATALOG_UNVERIFIED"))
LAN_HOST="xwing"
LAN_STATE="/home/scott/.local/state/colibri-glm53-observer-xwing/state.json"
TAIL_STATE=Path("/home/scott/.local/state/colibri-glm53-observer/state.json")

@dataclass(frozen=True)
class Witness:
    host:str
    path_kind:str
    remote_state:str
    alarm:str
    last_epoch_s:int
    consecutive_failures:int
    decision:str

def parse(payload:bytes|str, *,host:str,path_kind:str,now_s:int)->Witness:
    if host not in OWNERS or path_kind not in {"LAN","TAILNET"}:
        raise ValueError("NO_GO unexpected witness host or path")
    if not isinstance(payload,(str,bytes)) or len(payload)>MAX_STATE:
        raise ValueError("NO_GO invalid witness size")
    data=json.loads(payload)
    if not isinstance(data,dict) or data.get("schema")!=SCHEMA:
        raise ValueError("NO_GO unexpected witness schema")
    epoch=data.get("last_epoch_s")
    failures=data.get("consecutive_failures")
    if (type(epoch) is not int or type(failures) is not int or
        failures<0 or failures>1000000 or epoch<=0 or
        epoch>now_s or now_s-epoch>MAX_AGE_SECONDS):
        raise ValueError("NO_GO stale or invalid witness time")
    remote=data.get("remote_state")
    if (remote not in STATUS or data.get("decision")!="HOLD" or
        data.get("eligible_for_dispatch") is not False or
        data.get("failover_authorized") is not False or
        data.get("model_loaded_proven") is not False):
        raise ValueError("NO_GO unexpected model authority or status")
    alarm=data.get("alarm")
    if alarm not in {"REMOTE_UNREACHABLE",
                     "REMOTE_UNREACHABLE_PENDING_CONFIRMATION",
                     "MODEL_GENERATION_UNVERIFIED"}:
        raise ValueError("NO_GO invalid incident alarm")
    if alarm=="REMOTE_UNREACHABLE" and (
        remote not in {"UNREACHABLE","HEALTH_FAILED"} or failures<2):
        raise ValueError("NO_GO contradictory confirmed alarm")
    return Witness(host,path_kind,remote,alarm,epoch,failures,"HOLD")

def correlate(witnesses:list[Witness],*,now_s:int)->dict:
    result={"schema":1,"checked_at_epoch_s":now_s,
            "decision":"HOLD","eligible_for_dispatch":False,
            "failover_authorized":False,"model_loaded_proven":False,
            "corroboration":"NO_QUORUM",
            "witnesses":[w.__dict__ for w in witnesses]}
    if (len(witnesses)!=2 or
        {w.host for w in witnesses}!={"x1-370","xwing"} or
        {(w.host,w.path_kind) for w in witnesses}!={
            ("x1-370","TAILNET"),("xwing","LAN")} or
        any(now_s-w.last_epoch_s>MAX_AGE_SECONDS or
            w.last_epoch_s>now_s for w in witnesses)):
        result["reason"]="two fresh distinct nodes and network paths required"
        return result
    if abs(witnesses[0].last_epoch_s-witnesses[1].last_epoch_s)>MAX_CLOCK_SKEW_SECONDS:
        result["reason"]="independent observations too far apart"
        return result
    failed=[w for w in witnesses if
            w.alarm=="REMOTE_UNREACHABLE" and
            w.remote_state in {"UNREACHABLE","HEALTH_FAILED"}]
    if len(failed)==2:
        if all(w.remote_state=="UNREACHABLE" for w in failed):
            result["corroboration"]="JOYNER_UNREACHABLE_BOTH_NETWORK_PATHS"
            result["reason"]="both network observers confirmed transport failure"
        else:
            result["corroboration"]="JOYNER_HEALTH_OR_TRANSPORT_FAILURE_BOTH_PATHS"
            result["reason"]="both observers failed; at least one received unhealthy HTTP"
    elif len(failed)==1:
        result["corroboration"]="NETWORK_PATH_DISAGREEMENT"
        result["reason"]="only one network path reported confirmed failure; never auto-recover"
    else:
        result["corroboration"]="MODEL_GENERATION_UNVERIFIED"
        result["reason"]="transport observations cannot prove authenticated GLM generation"
    return result

def read_local(path:Path)->bytes:
    st=path.lstat()
    if (not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid() or
        st.st_mode&0o077 or st.st_nlink!=1 or st.st_size>MAX_STATE):
        raise ValueError("NO_GO unsafe local witness file")
    return path.read_bytes()

def read_xwing(*,timeout_seconds:float=7.0)->bytes:
    if not 1<=timeout_seconds<=10:raise ValueError("NO_GO invalid SSH deadline")
    p=subprocess.run([
        "ssh","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes",
        "-o","ConnectTimeout=4","-o","NumberOfPasswordPrompts=0",
        LAN_HOST,"head","-c",str(MAX_STATE+1),LAN_STATE
    ],capture_output=True,timeout=timeout_seconds,check=False)
    if p.returncode!=0 or len(p.stdout)>MAX_STATE:
        raise RuntimeError("NO_GO unavailable or oversized xwing witness")
    return p.stdout

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--local",type=Path,default=TAIL_STATE)
    ap.add_argument("--peer",type=Path,
                    help="offline test-only: existing peer receipt (otherwise strict SSH)")
    args=ap.parse_args()
    now=int(time.time())
    observed=[]
    errors=[]
    for host,network,loader in (
        ("x1-370","TAILNET",lambda:read_local(args.local)),
        ("xwing","LAN",lambda:read_local(args.peer) if args.peer else read_xwing())):
        try:
            observed.append(parse(loader(),host=host,path_kind=network,now_s=now))
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired,
                json.JSONDecodeError) as exc:
            errors.append({"host":host,"reason":type(exc).__name__})
    report=correlate(observed,now_s=now)
    if errors:report["witness_errors"]=errors
    print(json.dumps(report,sort_keys=True,indent=2))
    return 0 if len(observed)==2 else 2

if __name__=="__main__":
    raise SystemExit(main())
