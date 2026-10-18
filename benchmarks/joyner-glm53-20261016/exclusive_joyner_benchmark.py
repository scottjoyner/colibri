#!/usr/bin/env python3
"""Exclusive, reversible Joyner R9700 GLM A/B controller.

Explicitly stop ONLY two known non-admitted GPU canary systemd user services,
verify /dev/dri/renderD128 is free of other GPU users, and restore all
previously active services. Emergency systemd timer independently restores
services after 4 minutes if the orchestrator dies. Unknown users FAIL CLOSED.

No killall, no arbitrary process-name matching, no Docker/network/system
services changed. Use --dry-run to inspect without stopping anything.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from glm53_competitors import sample as sample_competitors
from glm53_quiesce import stop_units as stop_disposable_units

UNITS=("fleet-r9700-vulkan-canary.service","fleet-r9700-toolcall.service")
GPU_DEV=Path("/dev/dri/renderD128")
USER_RUN=Path("/run/user")/str(os.getuid())
LOCK=USER_RUN/"glm53-exclusive-benchmark.lock"
TARGET="100.83.215.83"
PORTS=(18320,18321)
RUNNER=Path("/home/joyner/glm53-native-runtime-shadow-20261017/run_joyner_native_tier_ab.py")
MODEL_BINARY=Path("/home/joyner/glm53-native-runtime-shadow-20261017/glm53-final-candidate-vulkan-20261017")
EXPERIMENT_BINARY=Path("/home/joyner/glm53-native-runtime-shadow-20261017/glm53-single-row-canary-20261017")
EXPERIMENT_SHA256="b2cf18d0b98da29f868818ad52bd84dd2d33c70d0448edb201863ede250c74a0"
RECEIPTS=Path("/home/joyner/glm53-native-runtime-shadow-20261017/physical_tier_ab_20261017")
RESTORE_SCRIPT=RUNNER.with_name("exclusive_joyner_restore.py")
RESTORE_AFTER="240s"

def call(*args,check=True,timeout=20):
    return subprocess.run(list(args),capture_output=True,text=True,check=check,timeout=timeout)

def unit_property(unit,prop):
    return call("systemctl","--user","show",unit,"-p",prop,"--value").stdout.strip()

def active(unit):
    return unit_property(unit,"ActiveState")=="active"

def gpu_users():
    if not GPU_DEV.is_char_device():raise RuntimeError("NO_GO expected GPU render node missing")
    if (Path("/sys/class/drm/renderD128/device").resolve()!=Path("/sys/class/drm/card1/device").resolve()):
        raise RuntimeError("NO_GO GPU DRM device mismatch")
    p=call("fuser","-a",str(GPU_DEV),check=False,timeout=8)
    if p.returncode not in (0,1):raise RuntimeError("NO_GO GPU ownership scan failed")
    return sorted(set(int(x) for x in re.findall(r"\b\d+\b",p.stdout)))

def health():
    import http.client
    status=[]
    for port in PORTS:
        con=http.client.HTTPConnection(TARGET,port,timeout=3)
        try:
            con.request("GET","/health")
            r=con.getresponse()
            r.read()
            status.append(r.status)
        finally:con.close()
    return status

def unexpected_competitors():
    # No silent kill of unknown GPU clients; identify precisely in evidence.
    known={int(unit_property(u,"MainPID")) for u in UNITS if active(u)}
    return sorted(set(gpu_users())-known)

def quiesce_background(*,known_pids:frozenset[int]=frozenset())->dict:
    """Actually stop disposable GPU/CPU/SSD competitors before admitting run."""
    activity=sample_competitors(
        duration_s=1.0,exclude_pids=frozenset((*known_pids,os.getpid())))
    gpu_extra=sorted(set(gpu_users())-known_pids)
    targets=sorted({int(p["pid"]) for p in activity["competitors"]}|set(gpu_extra))
    outcome=stop_disposable_units(targets)
    if outcome["status"] not in ("STOPPED",):
        raise RuntimeError("NO_GO cannot safely terminate competing workloads: "+
                           json.dumps(outcome,sort_keys=True))
    if targets:
        # Sensitive process command lines never enter the receipt. Log only
        # the exact disposable systemd units and PIDs we actually stopped.
        RECEIPTS.mkdir(parents=True,exist_ok=True)
        logfile=RECEIPTS/"exclusive_competitor_termination.jsonl"
        record={"utc":__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
                "outcome":outcome,
                "observed_cpu_or_io":activity["competitors"],
                "gpu_competitor_pids":gpu_extra}
        with logfile.open("a") as fd:
            fd.write(json.dumps(record,sort_keys=True)+"\n")
        print("EXCLUSIVE_TERMINATED_DISPOSABLE",json.dumps(outcome,sort_keys=True),flush=True)
    return {"targets":targets,"result":outcome,
            "background_nvme_bytes_s":activity["global_nvme_bytes_per_sec"]}

def preflight():
    if os.uname().nodename!="joyner" or os.getuid()==0:
        raise RuntimeError("NO_GO not the unprivileged Joyner user")
    original={u:active(u) for u in UNITS}
    pids={u:int(unit_property(u,"MainPID")) if original[u] else 0 for u in UNITS}
    unknown=unexpected_competitors()
    if unknown:raise RuntimeError("NO_GO unknown renderD128 GPU processes: "+str(unknown))
    if any(not value for value in original.values()):
        raise RuntimeError("NO_GO both known GPU canaries must be running for reversible baseline")
    if health()!=[200,200]:
        raise RuntimeError("NO_GO canary endpoints unhealthy before exclusive window")
    heavy=[]
    output=call("ps","-eo","pid=,pcpu=,rss=,comm=").stdout.splitlines()
    for row in output:
        f=row.split(maxsplit=3)
        if len(f)!=4:continue
        pid,cpu,rss,comm=f
        if float(cpu)>=80 or int(rss)>=2_000_000:
            if int(pid) not in pids.values() and comm not in ("ps",):
                heavy.append({"pid":int(pid),"cpu_percent":float(cpu),"rss_kib":int(rss),"comm":comm})
    if heavy:raise RuntimeError("NO_GO unknown heavy CPU/RAM competitors: "+json.dumps(heavy))
    activity=sample_competitors(duration_s=1.0,
                                exclude_pids=frozenset((os.getpid(),*pids.values())))
    if activity["competitors"]:
        raise RuntimeError("NO_GO CPU / SSD / RAM competitors: "+json.dumps(activity["competitors"]))
    if activity["global_nvme_bytes_per_sec"]>=32*1024*1024:
        raise RuntimeError("NO_GO NVMe background traffic high before test: "+str(activity["global_nvme_bytes_per_sec"]))
    return {"originally_running":original,"canary_pids":pids,
            "initial_gpu_users":gpu_users(),
            "health":health(),"unknown_gpu_users":unknown,
            "heavy_unknown":heavy,"cpu_io_baseline":activity}

def restart_and_verify(original):
    to_start=[u for u,was_running in original.items() if was_running]
    if to_start:call("systemctl","--user","start",*to_start,timeout=25)
    for _ in range(25):
        try:
            if all(active(u)==old for u,old in original.items()) and health()==[200,200]:
                return True
        except Exception:pass
        time.sleep(1)
    return False

def run_trial(label:str,*,binary:Path=MODEL_BINARY):
    fair="0" if label.startswith("exclusive4control-") else "1"
    stream="1" if label.startswith("exclusive4stream-") else "0"
    args=[sys.executable,str(RUNNER),"--binary",str(binary),
          "--out-dir",str(RECEIPTS),"--tier-gib","4","--rate","32",
          "--dense-host","0","--text-only","1","--fair-layers",fair,
          "--expert-gb","0.05","--label",label,
          "--single-row-stream",stream,"--isolated"]
    print("EXCLUSIVE_TRIAL",label,flush=True)
    p=subprocess.run(args,capture_output=True,text=True,timeout=125)
    print("EXCLUSIVE_TRIAL_EXIT",label,p.returncode,flush=True)
    if p.stdout:print(p.stdout[-1600:],flush=True)
    if p.returncode:
        raise RuntimeError("NO_GO benchmark trial "+label+": "+p.stderr[-600:])
    return json.loads((RECEIPTS/(label+".json")).read_text())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--single",choices=("control","fair","stream"),
                    help="run a single control or fair candidate instead of the pair")
    ap.add_argument("--compare-single-stream",action="store_true",
                    help="paired fair-share normal versus opt-in one-row streaming")
    ap.add_argument("--round-id",
                    help="optional 12-character lowercase hexadecimal receipt identity")
    a=ap.parse_args()
    if a.single and a.compare_single_stream:
        raise RuntimeError("NO_GO choose a single case or a pair, not both")
    if a.round_id and not re.fullmatch(r"[0-9a-f]{12}",a.round_id):
        raise RuntimeError("NO_GO round ID must be 12 lowercase hex characters")
    round_id=a.round_id or uuid.uuid4().hex[:12]
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open("w") as locked:
        try:fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("NO_GO GLM exclusive benchmark already active")
        # Validate output identity BEFORE any termination. Re-running a
        # completed case may never disrupt other jobs simply to return NO_GO.
        labels=([f"exclusive4{a.single}-{round_id}"] if a.single else
                [f"exclusive4fair-{round_id}",f"exclusive4stream-{round_id}"]
                if a.compare_single_stream else
                [f"exclusive4control-{round_id}",f"exclusive4fair-{round_id}"])
        print("EXCLUSIVE_ROUND_ID",round_id,"CASES",labels,flush=True)
        if not a.dry_run:
            for name in labels:
                if (RECEIPTS/(name+".json")).exists():
                    raise RuntimeError("NO_GO existing receipt "+name+"; never overwrite")
            if a.compare_single_stream:
                import hashlib
                if not EXPERIMENT_BINARY.is_file():
                    raise RuntimeError("NO_GO experimental binary absent")
                binary_sha=hashlib.sha256(EXPERIMENT_BINARY.read_bytes()).hexdigest()
                if binary_sha!=EXPERIMENT_SHA256:
                    raise RuntimeError("NO_GO unexpected experimental binary hash")
        # Claim the exclusive lock first, then remove ONLY leftover scopes
        # from this exact GLM benchmark namespace. Unknown workloads remain
        # untouched and cause a denial in preflight() instead.
        from exclusive_joyner_restore import stop_owned_scopes
        if not a.dry_run:
            stale=stop_owned_scopes()
            if stale:print("EXCLUSIVE_STOPPED_STALE_BENCH_SCOPES",stale,flush=True)
            # Known canary PIDs are intentionally stopped later, with restoration.
            known=frozenset(int(unit_property(u,"MainPID")) for u in UNITS if active(u))
            quiesce_background(known_pids=known)
        before=preflight()
        print("EXCLUSIVE_PREFLIGHT",json.dumps(before,sort_keys=True),flush=True)
        if a.dry_run:
            print("DRY_RUN_ONLY nothing stopped or benchmarked",flush=True)
            return 0
        tag="glm53-excl-restore-"+uuid.uuid4().hex[:8]
        # Register restoration BEFORE stopping anything. This timer survives
        # an abrupt SSH or Python exit. It is a backstop, not the normal path.
        if not RESTORE_SCRIPT.is_file():
            raise RuntimeError("NO_GO emergency restoration script missing")
        cmd=["systemd-run","--user","--on-active="+RESTORE_AFTER,
             "--unit="+tag,"--collect","--quiet",
             sys.executable,str(RESTORE_SCRIPT)]
        call(*cmd,timeout=12)
        original=before["originally_running"]
        stopped=[]
        succeeded=False
        try:
            call("systemctl","--user","stop",*UNITS,timeout=30)
            stopped=list(UNITS)
            for _ in range(10):
                if not any(active(u) for u in UNITS) and not gpu_users():break
                time.sleep(1)
            if any(active(u) for u in UNITS) or gpu_users():
                raise RuntimeError("NO_GO GPU still held after stopping exact canary units; "+str(gpu_users()))
            # An independently launched workload may start while services
            # are draining. Never knowingly benchmark with that contention.
            quiesce_background()
            quiet=sample_competitors(duration_s=1.0,
                                     exclude_pids=frozenset((os.getpid(),)))
            if quiet["competitors"]:
                raise RuntimeError("NO_GO residual CPU/SSD workload: "+json.dumps(quiet["competitors"]))
            if quiet["global_nvme_bytes_per_sec"]>=32*1024*1024:
                raise RuntimeError("NO_GO residual NVMe background traffic: "+str(quiet["global_nvme_bytes_per_sec"]))
            print("EXCLUSIVE_GPU_CLEAR",flush=True)
            print("EXCLUSIVE_CPU_SSD_CLEAR",json.dumps(quiet,sort_keys=True),flush=True)
            receipts=[]
            for label in labels:
                receipts.append(run_trial(label,binary=EXPERIMENT_BINARY
                                          if a.compare_single_stream else MODEL_BINARY))
            succeeded=True
            print("EXCLUSIVE_BENCH_RESULTS",json.dumps([
                {"name":r["mode"],"tok_s":r["tok_s"],
                 "gpu_served":r["gpu_served"],"expert_misses":r["expert_misses"],
                 "rss_gib":r["engine_rss_gib"]} for r in receipts]),flush=True)
        finally:
            restored=False
            try:
                # If interrupted mid-trial, its scope may outlive the runner.
                # Stop exact owned benchmark scopes BEFORE restoring services.
                from exclusive_joyner_restore import stop_owned_scopes
                cleaned=stop_owned_scopes()
                print("EXCLUSIVE_CLEANED_SCOPES",cleaned,flush=True)
                restored=restart_and_verify(original)
                print("EXCLUSIVE_RESTORE",restored,flush=True)
            finally:
                if restored:
                    # Cancel the backstop ONLY after both services return.
                    call("systemctl","--user","stop",tag+".timer",check=False,timeout=10)
                else:
                    print("NO_GO recovery failed; systemd restoration timer remains armed",file=sys.stderr,flush=True)
            if not restored:raise RuntimeError("NO_GO canary restore unsuccessful")
        return 0 if succeeded else 2

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print("NO_GO exclusive benchmark:",type(e).__name__,str(e),file=sys.stderr,flush=True)
        sys.exit(2)
