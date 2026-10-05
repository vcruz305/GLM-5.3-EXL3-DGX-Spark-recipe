#!/usr/bin/env python3
"""Fetch GLM-5.3's original BF16 lm_head.weight into a one-tensor safetensors file (stdlib only).

TensorFold's glm_moe_dsa engine reads lm_head.weight as BF16; the EXL3 pack stores an 8-bit EXL3 head instead. This
range-reads only that tensor (1.9 GB) out of zai-org/GLM-5.3's first shard at a pinned revision, instead of
downloading the 1.5 TB checkpoint, and writes it with a minimal header. The serve view maps lm_head.weight to it.

  python3 fetch_lm_head.py [OUT_DIR]          (default $HEAD_DIR, else ~/models/GLM-5.3-lm_head-bf16)

HEAD_REPO / HEAD_REV / HEAD_SHARD / HEAD_SHA / HEAD_BYTES come from env.sh. HF_TOKEN (or ~/.cache/huggingface/token)
is sent when present; zai-org/GLM-5.3 is not gated. The file is verified (size + sha256) before it is moved in place.
"""
import hashlib
import json
import os
import struct
import sys
import urllib.request

REPO = os.environ.get("HEAD_REPO", "zai-org/GLM-5.3")
REV = os.environ.get("HEAD_REV", "aca966e4e02791568aa6a4ced368624b3d897f42")
SHARD = os.environ.get("HEAD_SHARD", "model-00001-of-00141.safetensors")
WANT_SHA = os.environ.get("HEAD_SHA", "2df0f4a8469cf4a65295107130b8df4e5562e955ba1bd32f75b51f3e80ddb5f3")
WANT_BYTES = int(os.environ.get("HEAD_BYTES", "1903165544"))
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{SHARD}"


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.isfile(p) else None


def rng(a, b, tok):
    h = {"Range": f"bytes={a}-{b}"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    with urllib.request.urlopen(urllib.request.Request(URL, headers=h), timeout=600) as r:
        return r.read()


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "HEAD_DIR", os.path.expanduser("~/models/GLM-5.3-lm_head-bf16"))
    out = os.path.join(out_dir, "lm_head.safetensors")
    if os.path.isfile(out) and os.path.getsize(out) == WANT_BYTES:
        h = hashlib.sha256()
        with open(out, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        if h.hexdigest() == WANT_SHA:
            print(f"{out}: present, sha256 ok")
            return 0
        print(f"{out}: sha256 {h.hexdigest()} != {WANT_SHA}; fetching again")
    os.makedirs(out_dir, exist_ok=True)
    tok = token()
    n = struct.unpack("<Q", rng(0, 7, tok))[0]
    hdr = json.loads(rng(8, 8 + n - 1, tok))
    meta = hdr["lm_head.weight"]
    if meta["dtype"] != "BF16":
        print(f"lm_head.weight is {meta['dtype']} at {REPO}@{REV}, expected BF16", file=sys.stderr)
        return 1
    a, b = meta["data_offsets"]
    hdr2 = {"lm_head.weight": {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [0, b - a]}}
    h2 = json.dumps(hdr2).encode()
    h2 += b" " * ((8 - len(h2) % 8) % 8)
    tmp = out + ".part"
    sha = hashlib.sha256()
    with open(tmp, "wb") as f:
        for chunk in (struct.pack("<Q", len(h2)), h2):
            f.write(chunk)
            sha.update(chunk)
        pos, end, step = 8 + n + a, 8 + n + b, 64 << 20
        while pos < end:
            q = min(end, pos + step) - 1
            data = rng(pos, q, tok)
            if len(data) != q - pos + 1:
                print(f"short read at {pos}: {len(data)} bytes", file=sys.stderr)
                return 1
            f.write(data)
            sha.update(data)
            pos = q + 1
            print(f"\r{(pos - 8 - n - a) / 2**20:7.0f} / {(b - a) / 2**20:.0f} MiB", end="", flush=True)
    print()
    size, got = os.path.getsize(tmp), sha.hexdigest()
    if size != WANT_BYTES or got != WANT_SHA:
        print(f"{tmp}: {size} bytes sha256 {got}; expected {WANT_BYTES} / {WANT_SHA}. Left in place for inspection.",
              file=sys.stderr)
        return 1
    os.replace(tmp, out)
    with open(os.path.join(out_dir, "SOURCE.txt"), "w") as f:
        f.write(f"lm_head.weight {meta['shape']} {meta['dtype']} from https://huggingface.co/{REPO} revision {REV} "
                f"{SHARD}\nsha256 {got}\n")
    print(f"{out}: {size} bytes, sha256 {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
