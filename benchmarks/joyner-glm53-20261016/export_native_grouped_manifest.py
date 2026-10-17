#!/usr/bin/env python3
"""Private one-shot native pread/preadv manifest from pinned Joyner GLM holdout.

Reads real 62-shard headers and the frozen training/validation/evaluation split.
Writes only a 0600 benchmark manifest on Joyner. Never publishes model offsets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from benchmark_coalesced_complete_expert import source_index, frozen_workload
import benchmark_trace_predicted_reads as base

# Source index from PR13 is down/qs, gate/qs, up/qs.
# GLM53 native engine expects gate/qs, up/qs, down/qs.
NATIVE_PIECE_ORDER = (2, 3, 4, 5, 0, 1)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 16 <= args.rows <= 32:
        parser.error("bounded native shadow rows must be 16..32")
    if not args.out.parent.is_dir():
        parser.error("target private directory must already exist")
    if args.out.is_symlink() or args.out.exists():
        parser.error("NO_GO refusing existing output or symbolic link")
    base.safety()
    index = source_index()  # validates exact 62-shard pinned source
    batches, _, accuracy, heldout = frozen_workload(args.rows, gated=True)
    keys = [key for batch in batches for key in batch]
    if not 16 <= len(keys) <= 64 or len(set(keys)) < 2:
        parser.error("NO_GO unexpected manifest nomination count")
    lines = [str(len(keys))]
    for key in keys:
        parts = index[key]
        if len(parts) != 6:
            raise ValueError("NO_GO incomplete native expert")
        for p in NATIVE_PIECE_ORDER:
            path, off, size = parts[p]
            if path.parent != base.MODEL or not path.name.startswith("model-"):
                raise ValueError("NO_GO unexpected GLM source")
            shard = int(path.name.split("-")[1])
            if not 1 <= shard <= 62 or not 0 <= off or size <= 0:
                raise ValueError("NO_GO invalid native tensor source")
            lines.append(f"{shard} {off} {size}")
    payload = "\n".join(lines) + "\n"
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(payload)
    except BaseException:
        args.out.unlink(missing_ok=True)
        raise
    print(json.dumps({"mode":"private_native_shadow_export",
          "checkpoint":base.PIN,
          "heldout_sha256":hashlib.sha256(heldout.read_bytes()).hexdigest(),
          "rows":args.rows,"experts":len(keys),
          "native_tensor_order":"gate/up/down with 3 scale tensors",
          "file_mode":"0600","read_only_source":True,
          "accuracy":accuracy}, sort_keys=True))
    return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as error:
        print("NO_GO",type(error).__name__,str(error),file=sys.stderr)
        sys.exit(2)
