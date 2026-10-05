# Shared settings for GLM-5.3 on four DGX Sparks with TensorFold (TP=4). Sourced by setup.sh, serve.sh, rank.sh,
# chat.sh and drop-model-cache.sh. Every value can be overridden from the environment before calling those scripts;
# serve.sh forwards the ones you set to every Spark (FORWARD_VARS in serve.sh).
#
# PROFILE picks profiles/<name>.env first. Its values win over the defaults below; an exported variable wins over both.
#   fast-160k  (default)  --context 163840, whole bf16 KV cache on every rank (TF_GLM53_DCP=1). The measured fast path.
#   dcp4-262k             --context 262144 with decode context parallelism (TF_GLM53_DCP=4). Measured; slower, and
#                         another numeric path (its greedy ids differ from fast-160k's).
#   int4-262k             --context 262144, int4 latent KV cache, unsplit. PENDING VALIDATION: opt-in only
#                         (ALLOW_UNVALIDATED=1), needs the glm53-kv-int4 TensorFold branch (setup.sh builds it).

RECIPE_TP4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(dirname "$RECIPE_TP4_DIR")"

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*" >&2; }

PROFILE="${PROFILE:-fast-160k}"
[[ -f "$RECIPE_TP4_DIR/profiles/$PROFILE.env" ]] \
  || die "PROFILE=$PROFILE: no profiles/$PROFILE.env (choose fast-160k, dcp4-262k or int4-262k)"
# shellcheck source=profiles/fast-160k.env
source "$RECIPE_TP4_DIR/profiles/$PROFILE.env"

# ---- the runtime: TensorFold PR #159 + the measured loader fixes (vcruz305/TensorFold) ------------------------------
TF_REPO="${TF_REPO:-https://github.com/vcruz305/TensorFold.git}"
TF_BASE=689596d723115dc99d3c543e88847ebb93d00565         # drowzeys/TensorFold glm-moe-dsa-tp4 = ashhart/TensorFold#159
TF_MEASURED_REF=757a851a0ce72d5d0051f2cae6d9bc96e4a85a67  # vcruz305/TensorFold glm53-tp4-spark: the tree that was measured
TF_MEASURED_DIFF_SHA=1555d8ad577dccf99db990c90f9427b001ecdce91da05e23fcde01719ba4a7c7  # sha256 of `git diff BASE MEASURED`
TF_INT4_REF=47c7aa058f6d262b0a859f629c649780e0365515      # vcruz305/TensorFold glm53-kv-int4 (unvalidated, opt-in)
TF_REF="${TF_REF:-$TF_MEASURED_REF}"                      # profiles/int4-262k.env sets TF_INT4_REF

# ---- b12x RoCE one-shot reductions (local-inference-lab/b12x, Apache-2.0) -------------------------------------------
# PR #159's serving image uses b58f34e. PyPI b12x 1.3.0 predates the comm/roce module; do not pip install it.
B12X_REF=b58f34eaf978277621efced6678e6713fd7122e4
B12X_TARBALL_URL="https://codeload.github.com/local-inference-lab/b12x/tar.gz/$B12X_REF"
B12X_TARBALL_SHA="${B12X_TARBALL_SHA:-8cfd2d8bf09169d00a669f72f179c54bdfc49af35b05214a5e5f0fdb1d5779d8}"
B12X_DEPS="nvidia-cutlass-dsl==4.6.2 nvidia-cutlass-dsl-libs-base==4.6.2 nvidia-cutlass-dsl-libs-core==4.6.2 nvidia-cutlass-dsl-libs-cu13==4.6.2 apache-tvm-ffi==0.1.14.post1"

# ---- what gets served (Hugging Face, pinned revisions) --------------------------------------------------------------
PACK_REPO=vcruz305/GLM-5.3-EXL3-3.38bpw                  # gated: accept the access request on the model page first
PACK_REV=cc64e77b1b1744c7bc71e54c838727f0db0f1a36
PACK_CONFIG_SHA=f520ce2795fa4f38a835f84ad4d3db17983d63babb7bb16f6e23178b4612ba18
DRAFTER_REPO=incoai/GLM-5.3-DFlash2                       # CC BY-NC-ND 4.0: non-commercial use only
DRAFTER_REV=425aa615ce320caac34400208b30808c8f14f76c
HEAD_REPO=zai-org/GLM-5.3                                 # TensorFold reads a BF16 lm_head.weight: the original one
HEAD_REV=aca966e4e02791568aa6a4ced368624b3d897f42
HEAD_SHARD=model-00001-of-00141.safetensors
HEAD_BYTES=1903165544
HEAD_SHA=2df0f4a8469cf4a65295107130b8df4e5562e955ba1bd32f75b51f3e80ddb5f3  # of the file tools/fetch_lm_head.py writes
OFFICIAL_TEMPLATE_SHA=3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c
FIXED_TEMPLATE_SHA=2059ad4b073838cebd243d09b6633833e633edd866b95c549be25483909b00d0
TILES_SHA=db409731548ce676d14f29818ff689611202c27023f5cc4bcb93cbd871d6c84a

# ---- where things live on each Spark ----------------------------------------------------------------------------------
RECIPE_HOME="${RECIPE_HOME:-$HOME/glm53-tensorfold}"
VENV="${VENV:-$RECIPE_HOME/venv}"
TF_SRC_DIR="${TF_SRC_DIR:-$RECIPE_HOME/TensorFold-${TF_REF:0:12}}"
B12X_STAGE="${B12X_STAGE:-$RECIPE_HOME/b12x-${B12X_REF:0:12}}"
STATE_DIR="${STATE_DIR:-$RECIPE_HOME/state}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
PACK_DIR="${PACK_DIR:-$MODEL_ROOT/GLM-5.3-EXL3-3.38bpw}"
DRAFTER_DIR="${DRAFTER_DIR:-$MODEL_ROOT/GLM-5.3-DFlash2}"
HEAD_DIR="${HEAD_DIR:-$MODEL_ROOT/GLM-5.3-lm_head-bf16}"
VIEW_DIR="${VIEW_DIR:-$RECIPE_HOME/view}"                 # symlinks to the pack + lm_head + the fixed chat template
TILES="${TILES:-$RECIPE_TP4_DIR/tiles/tiles.json}"
HOSTS_FILE="${HOSTS_FILE:-$RECIPE_TP4_DIR/hosts}"

# ---- toolchain ----------------------------------------------------------------------------------------------------
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.14.1}"                 # the measured venv: torch 2.14.1+cu130
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
export CUDA_HOME

# ---- fabric (ConnectX-7) ------------------------------------------------------------------------------------------------
FABRIC_IFNAME="${FABRIC_IFNAME:-enp1s0f0np0}"             # the port that carries the fabric IPs in the hosts file
ROCE_HCA="${ROCE_HCA:-rocep1s0f0}"                        # its RDMA device (one rail, as measured)

# ---- the served configuration (values marked (measured) are the measured fast path) --------------------------------
TFS_ROCE="${TFS_ROCE:-1}"                                 # (measured) b12x RoCE one-shot decode reductions; 0 = NCCL
TFS_CONTEXT="${TFS_CONTEXT:-163840}"                      # --context (prompt + reply); profiles set it
TF_GLM53_DCP="${TF_GLM53_DCP:-1}"                         # 1: whole KV cache on every rank; 4: decode context parallel
TF_GLM53_CACHE_RESERVE_GB="${TF_GLM53_CACHE_RESERVE_GB:-6}"   # the engine's cache-guard spare (its default); keep it
TFS_KV_DTYPE="${TFS_KV_DTYPE:-bf16}"                      # (measured) bf16; int4/int8 only with PROFILE=int4-262k
TFS_PROMPT_ROWS="${TFS_PROMPT_ROWS:-2048}"                # (measured) prompt-chunk buffers (8192 does not fit memory)
TFS_VERIFY_ROWS="${TFS_VERIFY_ROWS:-8}"                   # (measured)
TFS_DFLASH_DEPTH="${TFS_DFLASH_DEPTH:-7}"                 # (measured) DFlash2 drafts per round, at most block - 1 = 7
TFS_DFLASH_CONFIDENCE="${TFS_DFLASH_CONFIDENCE:-0.60}"    # (measured) the engine default is 0.3
TFS_DRAFT_DEFAULT="${TFS_DRAFT_DEFAULT:-1}"               # (measured) requests draft with DFlash2 unless "draft": false
TFS_HEALTH="${TFS_HEALTH:-round}"                         # (measured) per-round all-rank b12x health check (~0.6 ms)
TFS_WARMUP="${TFS_WARMUP:-1}"                             # rank 0 warms up (serial + drafted, 16 tokens) before HTTP
TFS_THINKING="${TFS_THINKING:-1}"                         # 1: thinking on unless a request turns it off (measured)
TFS_MAX_TOKENS="${TFS_MAX_TOKENS:-4096}"                  # reply tokens when a request does not say
TFS_NAME="${TFS_NAME:-GLM-5.3-EXL3-3.38bpw}"              # model id in /v1/models and requests
TFS_ALIAS="${TFS_ALIAS:-glm-5.3}"
TFS_HTTP_HOST="${TFS_HTTP_HOST:-127.0.0.1}"               # rank 0; non-loopback needs TFS_API_KEY_FILE
TFS_HTTP_PORT="${TFS_HTTP_PORT:-8890}"
TFS_API_KEY_FILE="${TFS_API_KEY_FILE:-}"
TFS_MASTER_PORT="${TFS_MASTER_PORT:-29750}"               # rendezvous on rank 0; the RoCE setup's gloo uses +11

# ---- guards ---------------------------------------------------------------------------------------------------------
TFS_MIN_START_GIB="${TFS_MIN_START_GIB:-100}"             # MemAvailable each Spark needs before a load
TFS_MIN_AVAIL_GIB="${TFS_MIN_AVAIL_GIB:-2}"               # watchdog floor while serving: below it all four stop
TFS_MAX_SWAP_GROWTH_KB="${TFS_MAX_SWAP_GROWTH_KB:-0}"     # swap growth allowed over the watchdog's baseline (0: none)
TFS_WATCH_GRACE_S="${TFS_WATCH_GRACE_S:-45}"              # a rank gone while others run this long: stop all four
TFS_START_TIMEOUT_S="${TFS_START_TIMEOUT_S:-1800}"         # serve.sh up gives up after this (load is ~7-8 min)

# ---- what the launcher (lib/tf_serve_rank.py) reads, derived from the above ---------------------------------------------
TFS_CLONE="$TF_SRC_DIR/src"
TFS_VIEW="$VIEW_DIR"
TFS_DFLASH="$DRAFTER_DIR"
TFS_TILES="$TILES"
TFS_TILES_SHA="$TILES_SHA"
TFS_TEMPLATE_SHA="$FIXED_TEMPLATE_SHA"
TFS_CTL="$STATE_DIR/ctl"
TFS_LOGS="$STATE_DIR/logs"
TFS_IFNAME="$FABRIC_IFNAME"
TFS_HCA="$ROCE_HCA"
TFS_NCCL_LIB="${TFS_NCCL_LIB:-$(ls "$VENV"/lib/python3*/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null | head -n 1)}"

# ---- hosts file: "rank ssh_target fabric_ip [peer_ssh]", one line per rank ------------------------------------------
# Sets H_SSH[r], H_IP[r], H_PEER[r] (r = 0..3), TFS_MASTER (rank 0's fabric IP) and TFS_PEERS (ranks 1-3 as rank 0's
# watchdog reaches them over ssh: the 4th column, else the fabric IP). serve.sh reads the file where it runs and hands
# the same table to every Spark as HOSTS_INLINE ("rank ssh ip peer;..."), so the Sparks need no copy of the file.
read_hosts() {
  local table
  if [[ -n "${HOSTS_INLINE:-}" ]]; then
    table="${HOSTS_INLINE//;/$'\n'}"
  else
    [[ -f "$HOSTS_FILE" ]] || die "no hosts file at $HOSTS_FILE: cp $RECIPE_TP4_DIR/hosts.example $HOSTS_FILE and edit it"
    table="$(cat "$HOSTS_FILE")"
  fi
  H_SSH=(); H_IP=(); H_PEER=()
  local r s ip peer _rest n=0
  while read -r r s ip peer _rest; do
    [[ -z "${r:-}" || "$r" == \#* ]] && continue
    [[ "$r" =~ ^[0-3]$ ]] || die "$HOSTS_FILE: rank '$r' is not 0..3"
    [[ -n "${s:-}" && -n "${ip:-}" ]] || die "$HOSTS_FILE: rank $r needs 'rank ssh_target fabric_ip'"
    [[ -z "${H_SSH[$r]:-}" ]] || die "$HOSTS_FILE: rank $r listed twice"
    H_SSH[$r]="$s"; H_IP[$r]="$ip"; H_PEER[$r]="${peer:-$ip}"; n=$((n + 1))
  done <<< "$table"
  [[ $n == 4 && -n "${H_SSH[0]:-}" && -n "${H_SSH[1]:-}" && -n "${H_SSH[2]:-}" && -n "${H_SSH[3]:-}" ]] \
    || die "$HOSTS_FILE must list ranks 0, 1, 2 and 3 exactly once (found $n)"
  TFS_MASTER="${H_IP[0]}"
  TFS_PEERS="${H_PEER[1]} ${H_PEER[2]} ${H_PEER[3]}"
  HOSTS_INLINE="0 ${H_SSH[0]} ${H_IP[0]} ${H_PEER[0]};1 ${H_SSH[1]} ${H_IP[1]} ${H_PEER[1]};2 ${H_SSH[2]} ${H_IP[2]} ${H_PEER[2]};3 ${H_SSH[3]} ${H_IP[3]} ${H_PEER[3]}"
}

# ---- profile gate ---------------------------------------------------------------------------------------------------
check_profile() {
  if [[ "${PROFILE_VALIDATED:-1}" != 1 && "${ALLOW_UNVALIDATED:-0}" != 1 ]]; then
    die "PROFILE=$PROFILE is pending validation (no measured speed, memory or quality figure). Set ALLOW_UNVALIDATED=1 to run it anyway, or use the default profile"
  fi
  if [[ "${PROFILE_VALIDATED:-1}" != 1 ]]; then
    echo "warning: PROFILE=$PROFILE is PENDING VALIDATION: nothing about it has been measured on a Spark" >&2
  fi
  if [[ "$TFS_KV_DTYPE" != bf16 && "$TF_REF" == "$TF_MEASURED_REF" ]]; then
    die "TFS_KV_DTYPE=$TFS_KV_DTYPE needs the glm53-kv-int4 TensorFold branch (PROFILE=int4-262k sets it); the measured pin has no quantized cache"
  fi
  return 0
}

# ---- runtime check: the imported tree, not just a path ------------------------------------------------------------------
# Agents most often end up on another TensorFold (PyPI, a release tag, ashhart main) or a different venv. Check the
# checkout's commit, that it is clean, that it carries the measured loader fixes, and that b12x is the staged one.
verify_runtime() {
  local py="$VENV/bin/python"
  [[ -x "$py" ]] || die "no venv at $VENV. Run: bash tensorfold-four-spark-tp4/setup.sh"
  [[ -d "$TF_SRC_DIR/.git" ]] || die "no TensorFold checkout at $TF_SRC_DIR. Run: bash tensorfold-four-spark-tp4/setup.sh"
  local head; head="$(git -C "$TF_SRC_DIR" rev-parse HEAD)"
  [[ "$head" == "$TF_REF" ]] || die "TensorFold at $TF_SRC_DIR is ${head:0:12}, the recipe pins ${TF_REF:0:12}. Run setup.sh"
  [[ -z "$(git -C "$TF_SRC_DIR" status --porcelain --untracked-files=no)" ]] \
    || die "the TensorFold checkout $TF_SRC_DIR has local edits; the recipe serves the pinned tree only (git -C $TF_SRC_DIR stash)"
  if [[ "$TF_REF" == "$TF_MEASURED_REF" ]]; then
    local d; d="$(git -C "$TF_SRC_DIR" diff "$TF_BASE" "$TF_REF" | sha256sum | cut -d' ' -f1)"
    [[ "$d" == "$TF_MEASURED_DIFF_SHA" ]] || die "TensorFold diff vs PR #159 is $d, not the measured $TF_MEASURED_DIFF_SHA"
  fi
  "$py" - "$TFS_CLONE" "$TFS_KV_DTYPE" <<'PY' || die "the TensorFold / torch in $VENV is not the recipe runtime. Run: bash tensorfold-four-spark-tp4/setup.sh"
import importlib.util, os, sys
src, kv = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
spec = importlib.util.find_spec("tensorfold")
if spec is None or not os.path.realpath(spec.origin).startswith(os.path.realpath(src)):
    print(f"tensorfold resolves to {getattr(spec, 'origin', None)}, not {src}", file=sys.stderr); sys.exit(1)
fam = os.path.join(src, "tensorfold", "families", "glm_moe_dsa")
w = open(os.path.join(fam, "cuda", "weights.py")).read()
missing = [s for s in ("def drop_page_cache", "def _as_bf16", "def clear_cache") if s not in w]
if missing:
    print(f"{fam} lacks the measured loader fixes {missing} (PR #159 alone does not load this pack)", file=sys.stderr)
    sys.exit(1)
if kv != "bf16" and not os.path.isfile(os.path.join(fam, "cuda", "kvq.py")):
    print(f"TFS_KV_DTYPE={kv}: {fam} has no kvq.py (glm53-kv-int4 branch)", file=sys.stderr); sys.exit(1)
import torch
cuda = torch.version.cuda or ""
if not cuda.startswith("13."):
    print(f"torch {torch.__version__} is built for CUDA {cuda or 'none'}; GB10 needs a cu130 wheel", file=sys.stderr)
    sys.exit(1)
print(f"tensorfold at {src} (glm_moe_dsa + loader fixes{' + ' + kv + ' cache' if kv != 'bf16' else ''}), "
      f"torch {torch.__version__}", file=sys.stderr)
PY
  [[ -n "$TFS_NCCL_LIB" && -f "$TFS_NCCL_LIB" ]] || die "no libnccl.so.2 in $VENV (torch's nvidia-nccl wheel); TensorFold ctypes-loads it"
  if [[ "$TFS_ROCE" == 1 ]]; then
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$B12X_STAGE/site:$B12X_STAGE/site/nvidia_cutlass_dsl/dsl_packages" \
      "$py" -c 'from b12x.comm.roce import AllReduce' 2>/dev/null \
      || die "no b12x RoCE module staged at $B12X_STAGE (b12x @ ${B12X_REF:0:7}). Run setup.sh, or TFS_ROCE=0 for NCCL reductions"
    say "b12x @ ${B12X_REF:0:7} staged at $B12X_STAGE"
  fi
  [[ -f "$STATE_DIR/runtime-ref" && "$(cat "$STATE_DIR/runtime-ref")" == "$TF_REF" ]] \
    || echo "warning: setup.sh has not recorded a build of ${TF_REF:0:12} in $STATE_DIR/runtime-ref" >&2
  return 0
}

# ---- view check: what the server reads ----------------------------------------------------------------------------------
verify_view() {
  [[ -f "$VIEW_DIR/config.json" ]] || die "no serve view at $VIEW_DIR. Run setup.sh (or: bash tensorfold-four-spark-tp4/tools/make_view.sh)"
  local t; t="$(sha256sum "$VIEW_DIR/chat_template.jinja" | cut -d' ' -f1)"
  [[ "$t" == "$FIXED_TEMPLATE_SHA" ]] || die "$VIEW_DIR/chat_template.jinja is $t, not the fixed template. Re-run tools/make_view.sh"
  [[ "$(sha256sum "$(readlink -f "$VIEW_DIR/config.json")" | cut -d' ' -f1)" == "$PACK_CONFIG_SHA" ]] \
    || die "$VIEW_DIR/config.json is not the pinned pack's ($PACK_REPO @ ${PACK_REV:0:7})"
  grep -q '"lm_head.weight": "lm_head.safetensors"' "$VIEW_DIR/model.safetensors.index.json" \
    || die "$VIEW_DIR/model.safetensors.index.json does not map lm_head.weight. Re-run tools/make_view.sh"
  [[ "$(stat -L -c %s "$VIEW_DIR/lm_head.safetensors" 2>/dev/null)" == "$HEAD_BYTES" ]] \
    || die "$VIEW_DIR/lm_head.safetensors missing or not $HEAD_BYTES bytes. Run: $VENV/bin/python tensorfold-four-spark-tp4/tools/fetch_lm_head.py"
  [[ "$(sha256sum "$TILES" | cut -d' ' -f1)" == "$TILES_SHA" ]] || die "$TILES is not the pinned tile table"
  [[ -f "$DRAFTER_DIR/config.json" && -f "$DRAFTER_DIR/model.safetensors" ]] \
    || die "no DFlash2 drafter at $DRAFTER_DIR. Run setup.sh"
  local n; n="$(ls "$PACK_DIR"/model-*-of-00058.safetensors 2>/dev/null | wc -l)"
  [[ "$n" == 58 ]] || die "$PACK_DIR has $n of 58 weight shards. Re-run setup.sh (hf download resumes)"
  return 0
}
