#!/usr/bin/env python3
"""Emergency/normal cleanup for Joyner exclusive GLM53 benchmarks.

Only stops this controller's exact transient benchmark scope pattern, then
restores the two known non-admitted canaries. Never sends signals to unrelated
processes, Docker, networking, or the OS.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time

UNITS=("fleet-r9700-vulkan-canary.service",
       "fleet-r9700-toolcall.service")
SCOPE_RE=re.compile(r"^glm53-tier-ab-[0-9a-f]{12}\.scope$")

def run(*args,check=False,timeout=18):
    return subprocess.run(list(args),capture_output=True,text=True,
                          check=check,timeout=timeout)

def owned_scopes(listing:str)->list[str]:
    result=[]
    for line in listing.splitlines():
        for token in line.split():
            if SCOPE_RE.fullmatch(token):
                result.append(token)
                break
    return sorted(set(result))

def stop_owned_scopes()->list[str]:
    units=owned_scopes(run("systemctl","--user","list-units",
                           "--all","--type=scope","--no-legend",
                           "--plain",check=True).stdout)
    for unit in units:
        stopped=run("systemctl","--user","stop",unit,timeout=16)
        if stopped.returncode:
            # Exact temporary GLM scope only, never PID-pattern killall.
            run("systemctl","--user","kill","--signal=SIGKILL",unit,timeout=10)
            run("systemctl","--user","stop",unit,timeout=10)
    left=owned_scopes(run("systemctl","--user","list-units",
                          "--all","--type=scope","--no-legend",
                          "--plain",check=True).stdout)
    if left:raise RuntimeError("NO_GO owned GLM benchmark scopes remain: "+str(left))
    return units

def restore_canaries():
    r=run("systemctl","--user","start",*UNITS,timeout=30)
    if r.returncode:raise RuntimeError("NO_GO restore failed: "+r.stderr.strip())
    for _ in range(30):
        if all(run("systemctl","--user","is-active","--quiet",u).returncode==0
               for u in UNITS):
            return
        time.sleep(1)
    raise RuntimeError("NO_GO canary services failed to become active")

def main():
    if os.uname().nodename!="joyner" or os.getuid()==0:
        raise RuntimeError("NO_GO restore script requires unprivileged Joyner")
    stopped=stop_owned_scopes()
    restore_canaries()
    print("RESTORED_CANARIES after stopping owned GLM scopes:",stopped,flush=True)

if __name__=="__main__":
    try:main()
    except Exception as error:
        print("NO_GO emergency restore:",repr(error),file=sys.stderr,flush=True)
        sys.exit(2)
