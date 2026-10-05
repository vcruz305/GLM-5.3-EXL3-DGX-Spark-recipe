#!/usr/bin/env python3
"""Check that an EXL3 pack directory is complete without hashing it.

    python3 tools/check_pack.py <pack dir> [expected shard count]

Every model-*-of-N.safetensors shard must exist and be exactly as long as its own header says (8-byte header length +
header + the largest tensor end offset), and every file the index names must exist. A missing shard, a truncated
download or an interrupted copy fails; the bytes themselves are not hashed (that is hf download's job). stdlib only.
Exit 0 = complete, 1 = not.
"""
import json
import os
import struct
import sys


def shard_problem(path: str):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return f"{size} B, no safetensors header"
            (n,) = struct.unpack("<Q", raw)
            if n > 100 * 2**20 or 8 + n > size:
                return f"{size} B, header length {n} does not fit"
            header = json.loads(f.read(n))
    except (OSError, ValueError) as e:
        return f"unreadable: {e}"
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    want = 8 + n + end
    if size != want:
        return f"{size} B, header says {want} B ({'truncated' if size < want else 'too long'})"
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    d = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else None
    bad = []
    idx_path = os.path.join(d, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        print(f"{d}: no model.safetensors.index.json", file=sys.stderr)
        return 1
    files = sorted(set(json.load(open(idx_path))["weight_map"].values()))
    shards = sorted(f for f in os.listdir(d) if f.startswith("model-") and f.endswith(".safetensors"))
    if want is not None and len(shards) != want:
        bad.append(f"{len(shards)} of {want} model-*.safetensors shards present")
    for f in files:
        if not os.path.exists(os.path.join(d, f)):
            bad.append(f"{f}: named in the index, missing")
    for f in shards:
        p = shard_problem(os.path.join(d, f))
        if p:
            bad.append(f"{f}: {p}")
    if bad:
        for b in bad[:20]:
            print(f"{d}: {b}", file=sys.stderr)
        if len(bad) > 20:
            print(f"{d}: ... {len(bad) - 20} more", file=sys.stderr)
        return 1
    print(f"{d}: {len(shards)} shards complete (sizes match their headers)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
