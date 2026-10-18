#!/usr/bin/env python3
"""Read-only per-process CPU and block-I/O contention sampler for Joyner.
Unknown workloads are NEVER killed. The benchmark refuses contaminated runs.
No subprocesses are launched and no process commands/credentials are read.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import re
import time

MIB=1024*1024
DEFAULT_CPU_CORES=.80        # >80% of one core during the observation interval
DEFAULT_IO_BYTES_S=32*MIB   # persistent disk reads+writes above 32 MiB/s
DEFAULT_RSS_KIB=2*1024*1024
PINNED_NVME=Path("/sys/block/nvme0n1/stat")
# Linux block I/O stat fields: read sectors index 2, written sectors index 6.
# This also sees privileged processes whose per-PID /proc/io is unreadable.

def block_bytes(device_stat:Path=PINNED_NVME)->int:
    cols=device_stat.read_text().split()
    if len(cols)<11:raise ValueError("NO_GO invalid pinned NVMe stat")
    r,w=int(cols[2]),int(cols[6])
    if r<0 or w<0:raise ValueError("NO_GO negative disk sectors")
    return (r+w)*512

def block_rate(before:int,after:int,seconds:float)->float:
    if seconds<=0 or after<before:raise ValueError("NO_GO invalid disk read rate")
    return (after-before)/seconds

@dataclass(frozen=True)
class ProcessSample:
    pid:int
    starttime:int
    cpu_ticks:int
    io_bytes:int|None
    rss_kib:int
    comm:str
    cgroup:str

def process_sample(pid:int,root:Path=Path("/proc"))->ProcessSample|None:
    base=root/str(pid)
    try:
        raw=(base/"stat").read_text()
        mark=raw.rfind(") ")
        if mark<0:raise ValueError("invalid process stat")
        # Linux fields 3.. from raw[mark+2:] (utime=14, stime=15,
        # starttime=22). Distinguish PIDs reused between samples.
        fields=raw[mark+2:].split()
        if len(fields)<22:raise ValueError("short process stat")
        utime=int(fields[11]);stime=int(fields[12]);start=int(fields[19])
        comm=raw[raw.find("(")+1:mark][:48]
        mem=(base/"statm").read_text().split()
        rss_kib=int(mem[1])*os.sysconf("SC_PAGE_SIZE")//1024
        cgroup=(base/"cgroup").read_text()
        try:
            iom=(base/"io").read_text()
            kv=dict((a.strip(),int(b.strip())) for a,b in
                    (line.split(":",1) for line in iom.splitlines() if ":" in line))
            io_bytes=kv["read_bytes"]+kv["write_bytes"]
        except (OSError,KeyError,ValueError):
            io_bytes=None
        return ProcessSample(pid,start,utime+stime,io_bytes,rss_kib,comm,cgroup)
    except FileNotFoundError:
        return None     # naturally exited before the sample completed
    except (OSError,ValueError,IndexError):
        return None     # unreadable process; separately counted in scan()

def scan(root:Path=Path("/proc"))->tuple[dict[int,ProcessSample],int]:
    out={}
    unknown=0
    for p in root.iterdir():
        if not p.name.isdecimal():continue
        try:item=process_sample(int(p.name),root)
        except OSError:item=None
        if item is None:
            unknown+=1
        else:
            out[item.pid]=item
    return out,unknown

def observe(before:dict[int,ProcessSample],after:dict[int,ProcessSample],
            seconds:float,*,excluded_scope:str|None=None,
            exclude_pids:frozenset[int]=frozenset(),
            cpu_cores:float=DEFAULT_CPU_CORES,
            io_bytes_s:int=DEFAULT_IO_BYTES_S,
            rss_kib:int=DEFAULT_RSS_KIB)->list[dict]:
    if seconds<=0:raise ValueError("nonpositive sample period")
    clk=os.sysconf("SC_CLK_TCK")
    if clk<=0:raise ValueError("invalid CPU clock")
    result=[]
    for pid,end in after.items():
        if pid in exclude_pids:continue
        if excluded_scope and f"/{excluded_scope}" in end.cgroup:continue
        start=before.get(pid)
        if not start or start.starttime!=end.starttime:continue
        ticks=max(0,end.cpu_ticks-start.cpu_ticks)
        cpu=ticks/clk/seconds
        if end.io_bytes is not None and start.io_bytes is not None:
            io=max(0,end.io_bytes-start.io_bytes)/seconds
        else:
            io=None
        # An idle memory-heavy task still competes for scarce RAM, but
        # ordinary OS and desktop processes need not be terminated.
        reasons=[]
        if cpu>=cpu_cores:reasons.append("cpu")
        if io is not None and io>=io_bytes_s:reasons.append("disk_io")
        if end.rss_kib>=rss_kib:reasons.append("memory")
        if reasons:
            result.append(dict(pid=pid,comm=end.comm,
                               cpu_cores=round(cpu,3),
                               disk_bytes_per_sec=round(io) if io is not None else None,
                               rss_kib=end.rss_kib,reasons=reasons))
    return sorted(result,key=lambda e:e["pid"])

def sample(duration_s:float=1.,*,root:Path=Path("/proc"),
           excluded_scope:str|None=None,
           exclude_pids:frozenset[int]=frozenset())->dict:
    a,unread_before=scan(root)
    disk_before=block_bytes()
    start=time.monotonic()
    time.sleep(duration_s)
    b,unread_after=scan(root)
    disk_after=block_bytes()
    seconds=time.monotonic()-start
    return dict(duration_s=round(seconds,3),
                unknown_pid_reads=max(unread_before,unread_after),
                unreadable_process_io=sum(v.io_bytes is None for v in b.values()),
                checked_pid_count=len(b),
                global_nvme_bytes_per_sec=round(block_rate(disk_before,disk_after,seconds)),
                competitors=observe(a,b,seconds,
                                    excluded_scope=excluded_scope,
                                    exclude_pids=exclude_pids))
