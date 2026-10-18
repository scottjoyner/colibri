#!/usr/bin/env python3
"""Stop *verified disposable* user units competing with a GLM benchmark.

A unit is disposable only if it uses the explicit exact unit namespace
<glm53|colibri|opencode|hermes>-disposable-<12 hex>.service/.scope.
Never kill by process
name, substring, PID alone, arbitrary service, or permissions escalation.
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import subprocess
import time

UNIT=re.compile(r"^(?:glm53|colibri|opencode|hermes)-disposable-[0-9a-f]{12}\.(?:scope|service)$")
PROTECTED=frozenset((
    "fleet-r9700-vulkan-canary.service","fleet-r9700-toolcall.service",
    "ssh.service","sshd.service","tailscaled.service","docker.service",
))

def process_unit(pid:int,root:Path=Path("/proc"),uid:int|None=None)->str|None:
    if uid is None:uid=os.getuid()
    if pid<=1:return None
    try:
        status=(root/str(pid)/"status").read_text()
        m=re.search(r"^Uid:\s+(\d+)\s+",status,re.M)
        if not m or int(m.group(1))!=uid:return None
        lines=(root/str(pid)/"cgroup").read_text().splitlines()
    except (OSError,ValueError):return None
    for line in lines:
        for component in line.split(":",2)[-1].split("/"):
            if UNIT.fullmatch(component) and component not in PROTECTED:
                return component
    return None

def stop_units(pids:list[int]|set[int], *,
               root:Path=Path("/proc"),
               execute:bool=True,
               timeout_s:float=9.0)->dict:
    """Refuse unknown competing processes; never partially modify on mixed set.

    A positive result confirms the original PIDs are gone or no longer members
    of the approved workload unit. The scope/unit is stopped as a whole to
    prevent orphan descendants being left competing on GPU/SSD.
    """
    claimed={}
    untrusted=[]
    for pid in sorted(set(int(x) for x in pids)):
        if pid<=1 or pid==os.getpid():continue
        if not (root/str(pid)).exists():continue
        unit=process_unit(pid,root)
        if unit is None:untrusted.append(pid)
        else:claimed[pid]=unit
    if untrusted:
        return {"status":"HOLD","unknown_pids":untrusted,"stopped_units":[],
                "approved_pids":claimed}
    units=sorted(set(claimed.values()))
    if not execute:
        return {"status":"DRY_RUN","unknown_pids":[],"stopped_units":[],
                "would_stop_units":units,"approved_pids":claimed}
    stopped=[]
    for unit in units:
        # Revalidate the exact PID->unit relationship immediately before stop:
        # avoid PID reuse and stale process-identity assumptions.
        if not any(process_unit(pid,root)==unit for pid,u in claimed.items() if u==unit):
            continue
        p=subprocess.run(["systemctl","--user","stop",unit],
                         capture_output=True,text=True,timeout=12)
        if p.returncode:
            return {"status":"HOLD","unknown_pids":[],"stopped_units":stopped,
                    "failed_unit":unit,"failure":"user_unit_stop_failed"}
        stopped.append(unit)
    deadline=time.monotonic()+timeout_s
    while time.monotonic()<deadline:
        remaining=[pid for pid,u in claimed.items() if process_unit(pid,root)==u]
        if not remaining:
            return {"status":"STOPPED","unknown_pids":[],"stopped_units":stopped,
                    "approved_pids":claimed}
        time.sleep(.15)
    return {"status":"HOLD","unknown_pids":[],"stopped_units":stopped,
            "failed_unit":"verified_PIDs_still_alive"}

def candidate_pids(activity:dict)->list[int]:
    return sorted(set(
        [int(c["pid"]) for c in activity.get("competitors",[])]
        +[int(pid) for pid in activity.get("gpu_users",[])]
        +[int(pid) for pid in activity.get("cpu_or_ssd",[])]))
