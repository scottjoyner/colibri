#!/usr/bin/env python3
"""Fail-closed isolated full-GLM native tier A/B; no production services touched.

A previous *no-import* GLM run succeeded at 0.630 tok/s, ~5.56GiB RSS.
Imported-host variant stalled Joyner. This runner explicitly disables imports,
swap, warmstarts, trace writes and repeat admission. It rejects unexpected host
or staging volume, checks both production endpoints throughout, and terminates
the exact transient systemd unit on resource pressure. Results are not release
approval.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

STAGE = Path("/srv/joyner-gen5/colibri-fork-dense-stage")
MODEL = Path("/srv/joyner-gen5/models/glm53_colibri_i4")
STAGE_UUID = "b731381e-20b2-4c44-84a2-251b73970db1"
EXPECTED_GPU = "0x7551"
HOST_LIMIT = 6*1024**3
# Abort *before* approaching the earlier 198MiB near-exhaustion state.
MIN_AVAILABLE_KIB = 900000
MAX_PSI_FULL_10 = 4.0
MAX_PSI_SOME_10 = 8.0
MIN_GPU_FREE_BYTES = 4*1024**3
EXPECTED_GREEDY = "16 11 21 13 17 11 21 13"
MAX_SECONDS_PER_RUN = 100.0
RECENT_PSI_OBSERVATIONS = 2
PORTS = (18320,18321)
TARGET = "100.83.215.83"

def check_endpoint(port:int):
    conn=http.client.HTTPConnection(TARGET,port,timeout=3)
    try:
        conn.request("GET","/health")
        r=conn.getresponse()
        r.read()
        return r.status==200
    finally:conn.close()

def pressure():
    meminfo=Path("/proc/meminfo").read_text()
    m=re.search(r"^MemAvailable:\s+(\d+)\s+kB$",meminfo,re.M)
    if not m:raise RuntimeError("NO_GO MemAvailable unavailable")
    p={}
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        cols=line.split()
        p[cols[0]]={k:float(v) for k,v in (s.split("=",1) for s in cols[1:])}
    sample={"mem_available_kib":int(m[1]),
            "full_avg10":p["full"]["avg10"],
            "some_avg10":p["some"]["avg10"]}
    vram=Path("/sys/class/drm/card1/device/mem_info_vram_used")
    vram_total=Path("/sys/class/drm/card1/device/mem_info_vram_total")
    if not (vram.is_file() and vram_total.is_file()):
        raise RuntimeError("NO_GO R9700 VRAM telemetry missing")
    sample["gpu_vram_used_bytes"]=int(vram.read_text())
    sample["gpu_vram_total_bytes"]=int(vram_total.read_text())
    if (sample["gpu_vram_total_bytes"]<=0 or sample["gpu_vram_used_bytes"]<0 or
        sample["gpu_vram_used_bytes"]>sample["gpu_vram_total_bytes"]):
        raise RuntimeError("NO_GO invalid R9700 VRAM telemetry")
    return sample

def pressure_violation(reading):
    if reading["mem_available_kib"]<MIN_AVAILABLE_KIB:
        return "host_memory_floor"
    if reading["full_avg10"]>=MAX_PSI_FULL_10:
        return "full_PSI"
    if reading["some_avg10"]>=MAX_PSI_SOME_10:
        return "some_PSI"
    if (reading["gpu_vram_total_bytes"]-reading["gpu_vram_used_bytes"]<
        MIN_GPU_FREE_BYTES):
        return "gpu_vram_floor"
    return None

def validate(binary:Path,output:Path):
    if os.uname().nodename!="joyner":raise RuntimeError("NO_GO not Joyner")
    if not binary.is_file() or not os.access(binary,os.X_OK):raise RuntimeError("NO_GO binary absent")
    if not MODEL.is_dir() or MODEL.is_symlink():raise RuntimeError("NO_GO checkpoint missing")
    if not STAGE.is_dir() or STAGE.is_symlink():raise RuntimeError("NO_GO staging path missing")
    if output.exists() or output.is_symlink():raise RuntimeError("NO_GO output path exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    if not os.access(output.parent,os.W_OK):raise RuntimeError("NO_GO output unwritable")
    mount=subprocess.run(["findmnt","-no","UUID","-T",str(STAGE)],
                         check=True,capture_output=True,text=True).stdout.strip()
    if mount!=STAGE_UUID:raise RuntimeError("NO_GO wrong SSD stage volume")
    gpu=Path("/sys/class/drm/card1/device/device").read_text().strip()
    if gpu!=EXPECTED_GPU:raise RuntimeError("NO_GO R9700 not present")
    if os.geteuid()==0:raise RuntimeError("NO_GO refuse root")
    if pressure()["mem_available_kib"]<3500000:
        raise RuntimeError("NO_GO Joyner has insufficient initial RAM headroom")
    for port in PORTS:
        if not check_endpoint(port):raise RuntimeError("NO_GO unhealthy existing inference endpoint")

def run_trial(binary:Path,output:Path,label:str,tier:float,rate:int,dense_host:int,text_only:int,fair_layers:int,expert_gb:float):
    unit="glm53-tier-ab-"+uuid.uuid4().hex[:12]
    logfile=output.parent/(label+".log")
    metrics=output.parent/(label+"_pressure.jsonl")
    # Unique exact transient unit; TERM followed by KILL on watchdog expiry.
    env={**os.environ,
       "GLM53_BOUNDED_DENSE":"1","GLM53_DENSE_SPILL_DIR":str(STAGE),
       "GLM53_BITS":"4","GLM53_EXPERT_GB":str(expert_gb),"GLM53_MAXT":"64",
       "COLI_VULKAN":"1","COLI_VK_DEV":"1","COLI_VK_CHAIN":"1",
       "COLI_VK_DENSE_HOST":str(dense_host),"COLI_VK_TIER_GB":str(tier),
       "COLI_VK_TIER_RATE":str(rate),"COLI_VK_TIER_WARM":"0",
       "COLI_VK_TIER_FILL":"0","COLI_VK_STAGED":"1",
       "COLI_VK_UP_IMPORT":"0","COLI_VK_UP_POOL":"0",
       "COLI_VK_TIER_REPEAT_ADMIT":"0","GLM53_COALESCED_EXPERT_IO":"vector",
       "GLM53_TEXT_ONLY":str(text_only),
       "COLI_VK_TIER_FAIR_LAYERS":str(fair_layers),
       "USAGE_SAVE":"0","KVSAVE":"0","OMP_NUM_THREADS":"2"}
    env.pop("GLM53_HOST_IMPORT_EXPERIMENT",None)
    cmd=["systemd-run","--user","--scope","--unit",unit,"--collect","--quiet",
         "-p",f"MemoryMax={HOST_LIMIT}","-p","MemoryHigh=5800M",
         "-p","MemorySwapMax=0","-p","CPUWeight=20",
         "-p","IOWeight=20","/usr/bin/timeout","100s",
         "/usr/bin/env",*["%s=%s"%(k,env[k]) for k in (
            "GLM53_BOUNDED_DENSE","GLM53_DENSE_SPILL_DIR","GLM53_BITS",
            "GLM53_EXPERT_GB","GLM53_MAXT","COLI_VULKAN","COLI_VK_DEV",
            "COLI_VK_CHAIN","COLI_VK_DENSE_HOST","COLI_VK_TIER_GB",
            "COLI_VK_TIER_RATE","COLI_VK_TIER_WARM","COLI_VK_TIER_FILL",
            "COLI_VK_STAGED","COLI_VK_UP_IMPORT","COLI_VK_UP_POOL",
            "COLI_VK_TIER_REPEAT_ADMIT","GLM53_COALESCED_EXPERT_IO",
            "GLM53_TEXT_ONLY","COLI_VK_TIER_FAIR_LAYERS","USAGE_SAVE","KVSAVE","OMP_NUM_THREADS")],
         str(binary),"--model",str(MODEL),"--ids","11,21,13","--greedy","8"]
    # Capture the process output directly rather than requiring journal access.
    events=[]
    bad_streak=0
    start=time.monotonic()
    result={"mode":label,"tier_gib":tier,"rate":rate,"dense_host":dense_host,"text_only":text_only,"fair_layers":fair_layers,"expert_gb":expert_gb,"unit":unit,"binary_sha256":
             hashlib.sha256(binary.read_bytes()).hexdigest(),"completed":False}
    with logfile.open("x") as logs,metrics.open("x") as samples:
        p=subprocess.Popen(cmd,stdout=logs,stderr=subprocess.STDOUT,
                           env={**os.environ,"SYSTEMD_PAGER":""})
        try:
            while True:
                time.sleep(1.0)
                elapsed=time.monotonic()-start
                record={"elapsed_s":round(elapsed,2),**pressure()}
                events.append(record)
                samples.write(json.dumps(record)+"\n");samples.flush()
                bad=pressure_violation(record) is not None
                bad_streak=bad_streak+1 if bad else 0
                if bad_streak>=RECENT_PSI_OBSERVATIONS:
                    result["stop_reason"]="memory_pressure"
                if elapsed>MAX_SECONDS_PER_RUN:
                    result["stop_reason"]="watchdog_wall_limit"
                if int(elapsed)%4==0:
                    try:
                        if not all(check_endpoint(port) for port in PORTS):
                            result["stop_reason"]="production_health_lost"
                    except Exception:
                        result["stop_reason"]="production_health_check_failed"
                if result.get("stop_reason"):
                    subprocess.run(["systemctl","--user","stop",unit+".scope"],
                                   capture_output=True,timeout=10)
                    try:p.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        p.terminate()
                        try:p.wait(timeout=2)
                        except subprocess.TimeoutExpired:p.kill();p.wait()
                    break
                if p.poll() is not None:break
            if not result.get("stop_reason"):
                result["exit_code"]=p.wait(timeout=3)
            else:
                result["exit_code"]=p.returncode
        finally:
            # A controller exception or broken Fleet Commander connection
            # must not leave an unobserved full-model process running.
            if p.poll() is None:
                try:
                    subprocess.run(["systemctl","--user","stop",unit+".scope"],
                                   capture_output=True,timeout=10,check=False)
                finally:
                    try:p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.terminate()
                        try:p.wait(timeout=2)
                        except subprocess.TimeoutExpired:p.kill();p.wait()
    log=logfile.read_text(errors="replace")
    greedy=re.search(r"^greedy (.+)$",log,re.M)
    decode=re.search(r"^decode (\d+) token in ([\d.]+)s = ([\d.]+) tok/s",log,re.M)
    tier_match=re.search(r"device (\d+) of (\d+) routed experts",log)
    uploads=re.search(r"uploads (\d+) \(",log)
    miss=re.search(r"^experts hits (\d+) miss (\d+) bytes (\d+)",log,re.M)
    memory=re.search(r"RSS ([\d.]+) GiB",log)
    result.update({"duration_s":round(time.monotonic()-start,2),
       "max_gpu_vram_used_bytes":max((x.get("gpu_vram_used_bytes",0) for x in events),default=0),
       "min_available_kib":min(x["mem_available_kib"] for x in events),
       "max_psi_full_avg10":max(x["full_avg10"] for x in events),
       "max_psi_some_avg10":max(x["some_avg10"] for x in events),
       "greedy_ids":greedy[1] if greedy else None,
       "decode_s":float(decode[2]) if decode else None,
       "tok_s":float(decode[3]) if decode else None,
       "gpu_served":int(tier_match[1]) if tier_match else None,
       "gpu_routed":int(tier_match[2]) if tier_match else None,
       "gpu_uploads":int(uploads[1]) if uploads else None,
       "expert_misses":int(miss[2]) if miss else None,
       "expert_bytes":int(miss[3]) if miss else None,
       "engine_rss_gib":float(memory[1]) if memory else None})
    result["completed"]=(not result.get("stop_reason") and
                         result.get("exit_code")==0 and decode is not None and
                         int(decode[1])==8 and greedy is not None and
                         greedy[1].strip()==EXPECTED_GREEDY and
                         result["gpu_routed"]==3696)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--binary",required=True,type=Path)
    pa.add_argument("--out-dir",required=True,type=Path)
    pa.add_argument("--tier-gib",required=True,type=float)
    pa.add_argument("--rate",required=True,type=int)
    pa.add_argument("--label",required=True,choices=("baseline2","dense_host0","textonly2","textonly4","textonly8","textonly4control","textonly4fair","textonly8fair","textonly4faircache2"))
    pa.add_argument("--dense-host",required=True,type=int,choices=(0,1))
    pa.add_argument("--text-only",required=True,type=int,choices=(0,1))
    pa.add_argument("--fair-layers",type=int,default=0,choices=(0,1))
    pa.add_argument("--expert-gb",type=float,default=0.05)
    a=pa.parse_args()
    allowed={(2.0,16,1,0,0,"baseline2"),(2.0,16,0,0,0,"dense_host0"),
             (2.0,16,0,1,0,"textonly2"),(4.0,32,0,1,0,"textonly4"),
             (8.0,32,0,1,0,"textonly8"),
             (4.0,32,0,1,0,"textonly4control"),
             (4.0,32,0,1,1,"textonly4fair"),
             (8.0,32,0,1,1,"textonly8fair"),
             (4.0,32,0,1,1,"textonly4faircache2")}
    if (a.tier_gib,a.rate,a.dense_host,a.text_only,a.fair_layers,a.label) not in allowed:
        raise SystemExit("NO_GO only pinned tier+dense+vision+fair placements")
    if a.expert_gb != (1.25 if a.label=="textonly4faircache2" else 0.05):
        raise SystemExit("NO_GO non-pinned CPU expert-cache budget")
    report=a.out_dir/(a.label+".json")
    validate(a.binary,report)
    r=run_trial(a.binary,report,a.label,a.tier_gib,a.rate,a.dense_host,a.text_only,a.fair_layers,a.expert_gb)
    print(json.dumps(r,sort_keys=True),flush=True)
    if not r["completed"]:sys.exit(15)
    for port in PORTS:
        if not check_endpoint(port):
            print("NO_GO post-run endpoint unavailable",file=sys.stderr)
            sys.exit(17)

if __name__=="__main__":
    try:main()
    except BaseException as e:
        print("NO_GO runner",repr(e),file=sys.stderr)
        raise
