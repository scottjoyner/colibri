#!/usr/bin/env python3
"""One-shot bounded, non-admitting Joyner GLM observer for a systemd user timer.

Read-only network GETs; local append-only availability evidence only.
No health response authorizes model loading, routing, production failover,
service restart, or a promotion of the GLM-5.3 model.

Use an owned private state directory; fail closed on unsafe paths, files,
corrupt prior state, or exhausted journal capacity. There is intentionally
NO operator credential, webhook, or remote write API in this component.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
import time

from glm53_availability_probe import observe, fleet_observation, get_json

SCHEMA=1
MAX_JOURNAL=8*1024*1024
MAX_STATE=16384
OUTAGE_SAMPLES=2
REACHABLE_SAMPLES=3

def _private_directory(path:Path)->None:
    if not path.is_absolute() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("NO_GO unsafe observer state directory")
    parent=path.parent.stat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid!=os.getuid() or
        parent.st_mode&0o022):
        raise ValueError("NO_GO unsafe observer state parent")
    path.mkdir(mode=0o700,parents=False,exist_ok=True)
    info=path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or
        (info.st_mode & 0o077)):
        raise ValueError("NO_GO observer state directory not private")

def _private_regular(path:Path,*,max_size:int)->dict|None:
    if not path.exists() and not path.is_symlink():return None
    info=path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or
        info.st_mode&0o077 or info.st_nlink!=1 or info.st_size>max_size):
        raise ValueError("NO_GO unsafe state/evidence file")
    with path.open("r",encoding="utf-8") as handle:
        result=json.load(handle)
    if not isinstance(result,dict) or result.get("schema")!=SCHEMA:
        raise ValueError("NO_GO corrupted observer state")
    return result

def _atomic_state(path:Path,content:dict)->None:
    # Atomic replacement preserves the prior receipt on interrupted writes.
    payload=(json.dumps(content,sort_keys=True,separators=(",",":"))+"\n").encode()
    if len(payload)>MAX_STATE:raise ValueError("NO_GO oversized observer state")
    fd,name=tempfile.mkstemp(prefix=".glm53-observer-",dir=path.parent)
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,"wb") as out:
            out.write(payload);out.flush();os.fsync(out.fileno())
        fd=-1
        os.replace(name,path)
        dirfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dirfd)
        finally:os.close(dirfd)
    finally:
        if fd>=0:os.close(fd)
        if os.path.exists(name):os.unlink(name)

def decide(previous:dict|None,observation:dict,*,now_s:int)->dict:
    if type(now_s) is not int or now_s<=0:
        raise ValueError("NO_GO invalid observer time")
    if previous is not None:
        if (previous.get("schema")!=SCHEMA or
            type(previous.get("last_epoch_s")) is not int or
            now_s<=previous["last_epoch_s"]):
            raise ValueError("NO_GO nonmonotonic or corrupt prior observation")
    remote=observation.get("remote_state")
    if remote not in {"UNREACHABLE","HEALTH_FAILED","HOST_HTTP_ONLY",
                      "MODEL_NOT_LISTED","MODEL_CATALOG_UNVERIFIED"}:
        raise ValueError("NO_GO unknown remote state")
    down=remote in {"UNREACHABLE","HEALTH_FAILED"}
    prior_down=(previous or {}).get("consecutive_failures",0)
    prior_seen=(previous or {}).get("consecutive_reachable",0)
    failures=prior_down+1 if down else 0
    reachable=0 if down else prior_seen+1
    # Recovery is NOT equivalent to a working model! Do not clear a
    # confirmed remote outage to "available" on HTTP 200/catalog membership.
    # The observed transport may recover, but inference always remains HOLD.
    alarm="REMOTE_UNREACHABLE" if down and failures>=OUTAGE_SAMPLES else (
        "REMOTE_UNREACHABLE_PENDING_CONFIRMATION" if down else
        "MODEL_GENERATION_UNVERIFIED")
    return {"schema":SCHEMA,"last_epoch_s":now_s,
            "consecutive_failures":min(failures,1000000),
            "consecutive_reachable":min(reachable,1000000),
            "remote_state":remote,"alarm":alarm,
            "transport_recovered_samples":min(reachable,REACHABLE_SAMPLES),
            "model_loaded_proven":False,"eligible_for_dispatch":False,
            "failover_authorized":False,"decision":"HOLD"}

def record_once(directory:Path,observation:dict,*,now_s:int|None=None)->dict:
    if now_s is None:now_s=int(time.time())
    _private_directory(directory)
    lock=directory/"observer.lock"
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or
            info.st_mode&0o077 or info.st_nlink!=1):
            raise ValueError("NO_GO unsafe observer lock")
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=directory/"state.json"
        prev=_private_regular(state_path,max_size=MAX_STATE)
        state=decide(prev,observation,now_s=now_s)
        entry={"epoch_s":now_s,"remote_state":state["remote_state"],
               "alarm":state["alarm"],"consecutive_failures":state["consecutive_failures"],
               "local_host_healthy":bool(observation.get("local_host_healthy")),
               "local_default_route_present":bool(observation.get("local_gateway_advertised_route")),
               "model_loaded_proven":False,"decision":"HOLD"}
        line=(json.dumps(entry,separators=(",",":"),sort_keys=True)+"\n").encode()
        journal=directory/"observations.jsonl"
        if journal.is_symlink():raise ValueError("NO_GO symlink evidence journal")
        out=os.open(journal,os.O_APPEND|os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            inf=os.fstat(out)
            if (not stat.S_ISREG(inf.st_mode) or inf.st_uid!=os.getuid() or
                inf.st_mode&0o077 or inf.st_nlink!=1 or len(line)>MAX_JOURNAL):
                raise ValueError("NO_GO evidence journal invalid/full")
            if inf.st_size+len(line)>MAX_JOURNAL:
                # A bounded two-file history avoids unbounded disk use while
                # preserving ~weeks of minute-resolution receipts. Refuse
                # unsafe rotation destinations rather than overwrite aliases.
                previous=directory/"observations.previous.jsonl"
                if previous.is_symlink():raise ValueError("NO_GO symlink previous journal")
                if previous.exists():
                    old=previous.lstat()
                    if (not stat.S_ISREG(old.st_mode) or old.st_uid!=os.getuid()
                        or old.st_mode&0o077 or old.st_nlink!=1 or
                        old.st_size>MAX_JOURNAL):
                        raise ValueError("NO_GO unsafe previous journal")
                os.close(out)
                out=-1
                os.replace(journal,previous)
                out=os.open(journal,os.O_APPEND|os.O_WRONLY|os.O_CREAT|
                            os.O_EXCL|os.O_NOFOLLOW,0o600)
            if os.write(out,line)!=len(line):
                raise IOError("NO_GO evidence short write")
            os.fsync(out)
        finally:
            if out>=0:os.close(out)
        _atomic_state(state_path,state)
        return state
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--inference-base",required=True)
    ap.add_argument("--expected-model",required=True)
    ap.add_argument("--local-health-base",default="http://127.0.0.1:8899")
    ap.add_argument("--state-dir",required=True,type=Path)
    args=ap.parse_args()
    base=args.local_health_base.rstrip("/")
    remote=observe(args.inference_base,args.expected_model)
    observation=fleet_observation(
        get_json(base+"/health"),get_json(base+"/route"),remote)
    state=record_once(args.state_dir,observation)
    print(json.dumps({"observer":"read_only","remote_state":state["remote_state"],
                      "alarm":state["alarm"],"consecutive_failures":state["consecutive_failures"],
                      "model_loaded_proven":False,"decision":"HOLD",
                      "failover_authorized":False},sort_keys=True))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
