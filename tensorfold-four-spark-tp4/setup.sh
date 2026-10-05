#!/usr/bin/env bash
# One-time setup, run on EACH of the four Sparks (same clone path on all four is easiest):
#   - a venv with torch (cu130) and the pinned TensorFold fork commit (PR #159 + the measured loader fixes);
#   - b12x @ b58f34e staged next to it (RoCE one-shot decode reductions; PyPI's b12x lacks that module);
#   - downloads: the EXL3 pack (gated: accept access on its Hugging Face page, then `hf auth login`), the DFlash2
#     drafter (CC BY-NC-ND 4.0: non-commercial use only) and the original BF16 lm_head (range-read, 1.9 GB);
#   - the serve view (symlinks + BF16 lm_head + fixed chat template).
#
#   bash tensorfold-four-spark-tp4/setup.sh             build or update everything (idempotent; downloads resume)
#   bash tensorfold-four-spark-tp4/setup.sh --check     only verify an existing install (runtime + view)
#   SKIP_DOWNLOADS=1 bash ... setup.sh                  runtime only (copy the model directories yourself)
#   PROFILE=int4-262k bash ... setup.sh                 also builds the unvalidated glm53-kv-int4 checkout
# Overrides: see env.sh (RECIPE_HOME, MODEL_ROOT, PACK_DIR, TF_REF, CUDA_HOME, TORCH_SPEC, ...).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ "${1:-}" == "--check" ]]; then
  verify_runtime
  verify_view
  say "runtime and view OK (profile $PROFILE, TensorFold ${TF_REF:0:12}, context $TFS_CONTEXT, DCP $TF_GLM53_DCP, KV $TFS_KV_DTYPE)"
  exit 0
fi
[[ "$TF_REF" == "$TF_MEASURED_REF" || "$TF_REF" == "$TF_INT4_REF" ]] \
  || echo "warning: TF_REF=$TF_REF is neither the measured pin nor the int4 branch; nothing here was measured on it" >&2

# ---- 0. what the build needs ----------------------------------------------------------------------------------------
[[ "$(uname -m)" == aarch64 ]] || echo "warning: $(uname -m): this recipe was built and measured on DGX Spark (aarch64, GB10)" >&2
for c in git curl sha256sum "$PYTHON_BIN"; do command -v "$c" >/dev/null || die "$c not found"; done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "nvcc not found at $CUDA_HOME/bin/nvcc (TensorFold JIT-builds CUDA kernels; set CUDA_HOME, CUDA 13.x)"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ "$TFS_ROCE" == 1 ]]; then
  command -v gcc >/dev/null && [[ -f /usr/include/infiniband/verbs.h ]] \
    || die "b12x builds a small RDMA proxy with gcc + libibverbs: apt install gcc libibverbs-dev (or TFS_ROCE=0 for NCCL reductions)"
fi
mkdir -p "$RECIPE_HOME" "$STATE_DIR" "$TFS_LOGS" "$TFS_CTL" "$MODEL_ROOT"

# ---- 1. venv + torch ------------------------------------------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  say "creating venv $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip setuptools wheel ninja packaging "huggingface_hub[cli]"
inc="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
[[ -f "$inc/Python.h" ]] || die "no Python.h under $inc: Triton JIT-compiles against it (Ubuntu 24.04: apt install libpython3.12-dev)"
if ! "$PY" -c 'import sys, torch; sys.exit(0 if (torch.version.cuda or "").startswith("13.") else 1)' 2>/dev/null; then
  say "installing $TORCH_SPEC from $TORCH_INDEX_URL"
  "$PY" -m pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
fi
"$PY" -c 'import torch, triton; print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)'

# ---- 2. TensorFold fork at the pinned commit --------------------------------------------------------------------------
if [[ ! -d "$TF_SRC_DIR/.git" ]]; then
  say "cloning $TF_REPO into $TF_SRC_DIR"
  git clone -q "$TF_REPO" "$TF_SRC_DIR"
fi
git -C "$TF_SRC_DIR" fetch -q origin
git -C "$TF_SRC_DIR" cat-file -e "$TF_REF^{commit}" 2>/dev/null || git -C "$TF_SRC_DIR" fetch -q origin "$TF_REF"
git -C "$TF_SRC_DIR" checkout -q --detach "$TF_REF"
# Its dependencies, editable; the launcher also puts $TF_SRC_DIR/src first on sys.path.
"$PY" -m pip uninstall -y -q tensorfold >/dev/null 2>&1 || true
"$PY" -m pip install -q -e "$TF_SRC_DIR"
echo "$TF_REF" > "$STATE_DIR/runtime-ref"

# ---- 3. b12x @ b58f34e, staged with --target (the venv's site-packages is not touched) ------------------------------------
if [[ "$TFS_ROCE" == 1 ]]; then
  if [[ -f "$B12X_STAGE/.staged" && "$(cat "$B12X_STAGE/.staged")" == "$B12X_REF" ]]; then
    say "b12x @ ${B12X_REF:0:7} already staged at $B12X_STAGE"
  else
    say "staging b12x @ ${B12X_REF:0:7} at $B12X_STAGE"
    rm -rf "$B12X_STAGE/site"; mkdir -p "$B12X_STAGE/src" "$B12X_STAGE/site" "$B12X_STAGE/roce_cache" "$B12X_STAGE/compile_cache"
    tgz="$B12X_STAGE/src/b12x-$B12X_REF.tar.gz"
    [[ -s "$tgz" ]] || curl -fsSL -o "$tgz" "$B12X_TARBALL_URL"
    got="$(sha256sum "$tgz" | cut -d' ' -f1)"
    [[ "$got" == "$B12X_TARBALL_SHA" ]] \
      || die "b12x tarball sha256 $got != $B12X_TARBALL_SHA (GitHub regenerated the archive?). Inspect $tgz; B12X_TARBALL_SHA=$got to accept it"
    rm -rf "$B12X_STAGE/src/b12x-$B12X_REF" && tar -xzf "$tgz" -C "$B12X_STAGE/src"
    # cutlass-dsl's own deps the venv may lack (checked read-only against the venv)
    extra="$("$PY" - <<'PY'
import importlib.metadata as m
need = {"typing-extensions": "typing-extensions>=4.13", "protobuf": "protobuf>=6.30.2,<7",
        "nvidia-cuda-nvdisasm": "nvidia-cuda-nvdisasm>=13.3,<14"}
out = []
for k, spec in need.items():
    try:
        v = m.version(k)
        if k == "protobuf" and not (v.split(".")[0] == "6" and tuple(map(int, v.split(".")[:2])) >= (6, 30)):
            out.append(spec)
    except m.PackageNotFoundError:
        out.append(spec)
print(" ".join(out))
PY
)"
    # shellcheck disable=SC2086
    "$PY" -m pip install -q --target "$B12X_STAGE/site" --no-deps "$B12X_STAGE/src/b12x-$B12X_REF" $B12X_DEPS $extra
    # import check with no CUDA context, and prebuild the RDMA proxy (gcc + libibverbs) into the cache
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$B12X_STAGE/site:$B12X_STAGE/site/nvidia_cutlass_dsl/dsl_packages" \
    B12X_ROCE_CACHE_DIR="$B12X_STAGE/roce_cache" "$PY" - <<'PY' || die "b12x import check failed (see above)"
import inspect, torch
from b12x.comm.roce import AllReduce, _proxy
sig = inspect.signature(AllReduce.__init__)
need = ["exchange_group", "device", "max_size", "max_gather_bytes", "hca_names", "gid_index"]
assert all(k in sig.parameters for k in need), "AllReduce lacks the arguments TensorFold's roce.py passes"
assert "padded_gather" in inspect.signature(AllReduce.prepare).parameters
print("b12x RoCE import OK, proxy ABI", _proxy.load().roce_abi_version(), "cuda initialized:", torch.cuda.is_initialized())
PY
    echo "$B12X_REF" > "$B12X_STAGE/.staged"
  fi
fi

# ---- 4. downloads -------------------------------------------------------------------------------------------------------
HF="$VENV/bin/hf"
if [[ -z "${SKIP_DOWNLOADS:-}" ]]; then
  say "pack $PACK_REPO @ ${PACK_REV:0:7} -> $PACK_DIR (319 GB; gated: request access on the model page, then '$HF auth login')"
  "$HF" download "$PACK_REPO" --revision "$PACK_REV" --local-dir "$PACK_DIR" --max-workers "${HF_MAX_WORKERS:-8}" \
    || die "pack download failed. Re-run to resume. Gated repo: accept access on huggingface.co/$PACK_REPO and log in. If transfers stall, retry with HF_HUB_DISABLE_XET=1"
  say "drafter $DRAFTER_REPO @ ${DRAFTER_REV:0:7} -> $DRAFTER_DIR (CC BY-NC-ND 4.0: non-commercial use only; do not redistribute)"
  "$HF" download "$DRAFTER_REPO" --revision "$DRAFTER_REV" --local-dir "$DRAFTER_DIR" \
    || die "drafter download failed; re-run to resume"
  say "BF16 lm_head from $HEAD_REPO @ ${HEAD_REV:0:7} -> $HEAD_DIR (range read of one tensor, 1.9 GB)"
  HEAD_REPO="$HEAD_REPO" HEAD_REV="$HEAD_REV" HEAD_SHARD="$HEAD_SHARD" HEAD_SHA="$HEAD_SHA" HEAD_BYTES="$HEAD_BYTES" \
    "$PY" "$RECIPE_TP4_DIR/tools/fetch_lm_head.py" "$HEAD_DIR"
else
  say "SKIP_DOWNLOADS set: expecting $PACK_DIR, $DRAFTER_DIR and $HEAD_DIR to be in place"
fi

# ---- 5. serve view + verify ---------------------------------------------------------------------------------------------
bash "$RECIPE_TP4_DIR/tools/make_view.sh"
verify_runtime
verify_view
# downloads leave the pack in page cache, which GB10's cudaMemGetInfo does not count as free
bash "$RECIPE_TP4_DIR/drop-model-cache.sh" >/dev/null 2>&1 || true
cat >&2 <<EOF

Setup complete on $(hostname) (profile $PROFILE).
  runtime:  vcruz305/TensorFold ${TF_REF:0:12}  ($TF_SRC_DIR)
  b12x:     $( [[ "$TFS_ROCE" == 1 ]] && echo "${B12X_REF:0:7} at $B12X_STAGE" || echo "off (TFS_ROCE=0: NCCL reductions)" )
  venv:     $VENV
  view:     $VIEW_DIR

Repeat on the other three Sparks, then from the machine that holds the hosts file:
  bash tensorfold-four-spark-tp4/serve.sh preflight && bash tensorfold-four-spark-tp4/serve.sh up
EOF
