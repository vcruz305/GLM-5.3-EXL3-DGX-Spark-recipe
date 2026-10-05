#!/usr/bin/env bash
# Drop the clean page cache of the model files on THIS Spark. On GB10 unified memory, cached file pages count against
# what cudaMemGetInfo reports as free, and TensorFold's cache guard (runner._check_cache_fits) then refuses the
# context with "x GiB is free" while MemAvailable looks fine. No root needed: posix_fadvise(DONTNEED) per
# *.safetensors file (symlinks followed); only clean, unmapped pages go, so it is safe while nothing is loading.
#   bash tensorfold-four-spark-tp4/drop-model-cache.sh [dir ...]     (default: the pack, view, lm_head and drafter)
# serve.sh up runs it on every Spark before the ranks start.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
if [[ $# -eq 0 ]]; then set -- "$PACK_DIR" "$VIEW_DIR" "$HEAD_DIR" "$DRAFTER_DIR"; fi
python3 - "$@" <<'PY'
import os, sys
n = 0
for d in sys.argv[1:]:
    for root, _, files in os.walk(d, followlinks=True):
        for fn in files:
            if not fn.endswith(".safetensors"):
                continue
            try:
                fd = os.open(os.path.join(root, fn), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
m = {}
for ln in open("/proc/meminfo"):
    k, v = ln.split(":", 1)
    m[k] = int(v.split()[0])
print(f"{os.uname().nodename}: fadvised {n} files; MemFree {m['MemFree']/2**20:.1f} GiB, MemAvailable "
      f"{m['MemAvailable']/2**20:.1f} GiB, Cached {m['Cached']/2**20:.1f} GiB, swap used "
      f"{(m['SwapTotal']-m['SwapFree'])/1024:.1f} MiB")
PY
