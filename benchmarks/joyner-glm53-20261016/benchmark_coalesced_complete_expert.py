#!/usr/bin/env python3
"""Bounded read-only Joyner A/B: six expert tensors vs exactly contiguous extents.

No live router, model writes, GPU allocation, or speculative inference.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_trace_predicted_reads as base
import benchmark_complete_expert_qd2 as prior
from calibrated_prefetch_budget import fit
from replay_expert_prefetch import compile_stats, read_trace

BLOCK = 4096
EXPECTED = dict(prior.TENSORS)
MEMORY_LIMIT = 32 * 1024 * 1024

def source_index():
    """Reconstruct *unaligned* exact tensor bounds from 62 pinned shard headers."""
    if len(base.validate_model()) != 12384:
        raise ValueError("NO_GO source-index validation failed")
    manifest = json.loads((base.MODEL / ".converted.json").read_text())
    results = {}
    for shard in sorted(manifest):
        path = base.MODEL / shard
        st = path.stat()
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("NO_GO truncated safetensor header prefix")
            n = struct.unpack("<Q", raw)[0]
            if n > 50_000_000:
                raise ValueError("NO_GO header length")
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if ".mlp.experts." not in name:
                continue
            import re
            match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)", name)
            if match is None:
                continue
            layer, expert = map(int, match.groups()[:2])
            tensor_name = match.group(3)
            if tensor_name not in EXPECTED:
                continue
            low, high = meta["data_offsets"]
            start, size = 8+n+low, high-low
            if (size != EXPECTED[tensor_name] or start < 8+n
                    or start+size > st.st_size or
                    meta.get("dtype") != ("F32" if tensor_name.endswith(".qs") else "U8")):
                raise ValueError("NO_GO tensor bounds/dtype/size")
            key = (layer, expert)
            entry = results.setdefault(key, {})
            if tensor_name in entry:
                raise ValueError("NO_GO duplicated tensor")
            entry[tensor_name] = (path, start, size)
    if len(results) != 12384:
        raise ValueError("NO_GO incomplete model index")
    for key, row in results.items():
        if set(row) != set(EXPECTED):
            raise ValueError(f"NO_GO incomplete expert {key}")
    return {k: tuple(row[n] for n in EXPECTED) for k, row in results.items()}

def extents(parts, grouped):
    """Merge only byte-adjacent tensors on identical source file; no gap overfetch."""
    ordered = sorted(parts, key=lambda p: (str(p[0]), p[1]))
    previous_file = None
    previous_end = -1
    spans = []
    for path, off, size in ordered:
        if previous_file == path and off < previous_end:
            raise ValueError("NO_GO overlapping source tensors")
        if grouped and spans and previous_file == path and off == previous_end:
            spans[-1][2] = off + size
            spans[-1][3].append((path, off, size))
        else:
            spans.append([path, off, off+size, [(path, off, size)]])
        previous_file, previous_end = path, off+size
    if len(spans) > 6 or not spans:
        raise ValueError("NO_GO invalid expert segmentation")
    return spans

def open_checked(paths):
    """Read-only, no symlink traversal, pinned source descriptors."""
    handles = {}
    tails = {}
    snapshots = {}
    try:
        for path in paths:
            pre = path.stat()
            direct = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC | os.O_NOFOLLOW)
            tail = None
            try:
                tail = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                st = os.fstat(direct)
                if ((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) !=
                     (pre.st_dev, pre.st_ino, pre.st_size, pre.st_mtime_ns)):
                    raise ValueError("NO_GO source descriptor identity changed")
                ts = os.fstat(tail)
                if (ts.st_ino, ts.st_dev) != (st.st_ino, st.st_dev):
                    raise ValueError("NO_GO EOF descriptor source mismatch")
                handles[path], tails[path] = direct, tail
                snapshots[path] = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            except BaseException:
                os.close(direct)
                if tail is not None:
                    os.close(tail)
                raise
        return handles, tails, snapshots
    except BaseException:
        for fd in list(handles.values()) + list(tails.values()):
            os.close(fd)
        raise

def checked_read(spans, handles, tails, stats, *, verify=False):
    """Bounded mmap per extent; EOF tail is buffered only for final <4096B."""
    begun = time.perf_counter()
    hashes = []
    for path, start, end, parts in spans:
        st = os.fstat(handles[path])
        aligned, count, tail_at, tail_len = prior.safe_aligned_span(start, end-start, st.st_size)
        if count > MEMORY_LIMIT:
            raise ValueError("NO_GO oversized expert IO extent")
        with mmap.mmap(-1, count, access=mmap.ACCESS_WRITE) as b:
            got = os.preadv(handles[path], [b], aligned)
            if got != count:
                raise OSError("NO_GO short O_DIRECT read")
            extra = os.pread(tails[path], tail_len, tail_at) if tail_len else b""
            if len(extra) != tail_len:
                raise OSError("NO_GO short buffered EOF tail")
            if verify:
                raw = bytes(b) + extra
                for src, offset, size in parts:
                    logical = raw[offset-aligned:offset-aligned+size]
                    expected = os.pread(tails[src], size, offset)
                    if logical != expected:
                        raise ValueError("NO_GO grouped tensor payload mismatch")
                    hashes.append(hashlib.sha256(logical).hexdigest())
        stats["calls"] += 1
        stats["direct_bytes"] += count
        stats["tail_bytes"] += tail_len
        stats["tail_calls"] += bool(tail_len)
    return (time.perf_counter()-begun)*1000, hashes

def check_sources(handles, snapshots):
    for path, fd in handles.items():
        st = os.fstat(fd)
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != snapshots[path]:
            raise ValueError("NO_GO source changed during A/B")

def frozen_workload(rows, gated):
    names = sorted(base.PRIVATE.glob("glm53_native_expert_trace_20261016_run*.jsonl"))
    if len(names) != 6 or "run05" not in names[4].name or "run06_v2" not in names[5].name:
        raise ValueError("NO_GO routing splits not pinned")
    freq, cooc = compile_stats([read_trace(path) for path in names[:4]])
    calibration = fit(read_trace(names[4]),
        lambda event, history: base.build_policy("fusion", event, history, freq, cooc),
        threshold=.4)
    events = read_trace(names[5])[:rows]
    selected, accuracy = prior.get_event_plans(events, freq, cooc, gated, calibration)
    batches = [[(layer, p["expert"]) for p in items
                if p["tier"] == "unknown_disk_or_page_cache"]
               for layer, items in selected]
    return batches, selected, accuracy, names[5]

def physical_one_mode(selected, index, grouped):
    base.safety()
    paths = {part[0] for batch in selected for key in batch for part in index[key]}
    fd, tail, snaps = open_checked(paths)
    samples, steps = [], []
    physical = {"calls":0, "direct_bytes":0, "tail_bytes":0, "tail_calls":0}
    disk_before = prior.disk_sector_bytes()
    try:
        # Keep QD2, but isolate per-worker accounting to avoid shared counter races.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for pos, batch in enumerate(selected):
                if not batch:
                    continue
                start = time.perf_counter()
                def work(key):
                    counters = {"calls":0, "direct_bytes":0, "tail_bytes":0, "tail_calls":0}
                    elapsed, _ = checked_read(extents(index[key], grouped), fd, tail, counters)
                    return elapsed, counters
                vals = list(pool.map(work, batch))
                steps.append((time.perf_counter()-start)*1000)
                for elapsed, counters in vals:
                    samples.append(elapsed)
                    for k in physical:
                        physical[k] += counters[k]
                if pos % 16 == 0:
                    base.safety()
        check_sources(fd, snaps)
        disk_delta = prior.disk_sector_bytes() - disk_before
    finally:
        for h in list(fd.values()) + list(tail.values()):
            os.close(h)
    physical.update(experts=len(samples), sector_delta_bytes=disk_delta,
        expert_p50_ms=base.percentile(samples,.5),
        expert_p95_ms=base.percentile(samples,.95),
        expert_p99_ms=base.percentile(samples,.99),
        qd2_pair_p99_ms=base.percentile(steps,.99))
    if len(samples) != sum(len(batch) for batch in selected):
        raise ValueError("NO_GO missing read")
    return physical

def verify_layouts(index):
    grouped_counts = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    eof = None
    for key, parts in index.items():
        spans = extents(parts, True)
        grouped_counts[len(spans)].append(key)
        if eof is None and any(p[1]+p[2] == p[0].stat().st_size and p[0].stat().st_size % BLOCK
                               for p in parts):
            eof = key
    if sum(len(v) for v in grouped_counts.values()) != 12384:
        raise ValueError("NO_GO unclassified experts")
    verify_keys = [v[0] for v in grouped_counts.values() if v]
    if eof is not None and eof not in verify_keys:
        verify_keys.append(eof)
    paths = {p[0] for key in verify_keys for p in index[key]}
    fds, tails, snap = open_checked(paths)
    try:
        for key in verify_keys:
            grouped_stats = {"calls":0,"direct_bytes":0,"tail_bytes":0,"tail_calls":0}
            baseline_stats = dict(grouped_stats)
            _, sha_grouped = checked_read(extents(index[key], True), fds, tails, grouped_stats, verify=True)
            _, sha_base = checked_read(extents(index[key], False), fds, tails, baseline_stats, verify=True)
            if sha_grouped != sha_base:
                raise ValueError(f"NO_GO source tensor hash mismatch {key}")
        check_sources(fds, snap)
    finally:
        for fd in list(fds.values()) + list(tails.values()):
            os.close(fd)
    return {"layout_counts": {str(k):len(v) for k,v in grouped_counts.items() if v},
            "verified_layout_keys": [list(k) for k in verify_keys], "eof_checked":eof is not None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--gated", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--execute-io", action="store_true",
                    help="explicitly opt in to physical four-pass read benchmark")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not 16 <= args.rows <= 462: ap.error("rows must be 16..462")
    if args.execute_io and args.verify_only: ap.error("conflicting I/O modes")
    if args.execute_io:
        cgroups = [line for line in Path("/proc/self/cgroup").read_text().splitlines()
                   if line.startswith("0::")]
        if len(cgroups) != 1:
            raise ValueError("NO_GO expected exactly one cgroup v2 entry")
        scope = Path("/sys/fs/cgroup") / cgroups[0].split("::",1)[1].lstrip("/")
        cap = (scope / "memory.max").read_text().strip()
        swap = (scope / "memory.swap.max").read_text().strip()
        if cap == "max" or int(cap) > 512*1024*1024 or swap != "0":
            raise ValueError("NO_GO physical I/O requires <=512MiB memory and zero swap")
    base.safety()
    index = source_index()
    verified = verify_layouts(index)
    report = {"schema":1,"date":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "checkpoint":base.PIN,"read_only":True,"validation":verified}
    if args.execute_io:
        nominated, events, accuracy, trace = frozen_workload(args.rows,args.gated)
        result = []
        # AB/BA order reduces exposure to monotonic drive cache/wear or foreground load.
        for grouped in (False, True, True, False):
            result.append({"mode":"grouped" if grouped else "six",
                           **physical_one_mode(nominated,index,grouped)})
        report.update(rows=args.rows,gated=args.gated,accuracy=accuracy,
                      heldout_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                      nominated=sum(map(len,nominated)),observations=result,
                      note="No GPU upload, native layer timing or tokens/s measurement")
    payload = json.dumps(report,indent=2)+"\n"
    if args.out:
        args.out.write_text(payload)
    print(payload)

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("NO_GO:", type(e).__name__, str(e), file=sys.stderr)
        raise SystemExit(2)
