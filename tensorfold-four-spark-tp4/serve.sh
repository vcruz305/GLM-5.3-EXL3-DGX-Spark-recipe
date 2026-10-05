#!/usr/bin/env bash
# Serve GLM-5.3 (EXL3 3.38 bpw) on four DGX Sparks: TensorFold TP=4, one rank per Spark, an OpenAI-compatible /v1 API
# on rank 0 (http://127.0.0.1:8890/v1 there, model id GLM-5.3-EXL3-3.38bpw). This is the recommended way to run it.
# Run it where the hosts file is (a Spark or any machine with ssh to all four); every host-side action is rank.sh on
# that Spark, in the same recipe clone (REMOTE_REPO, default: this clone's path).
#
#   bash tensorfold-four-spark-tp4/serve.sh preflight   every Spark: same recipe commit, GPU idle, MemAvailable, ports,
#                                                       RoCE GID, runtime pin, view shas, launcher dry run
#   bash tensorfold-four-spark-tp4/serve.sh up          drop page cache -> watchdogs -> ranks 1,2,3 -> rank 0 -> wait
#                                                       for "serving" (load ~7 min; a first start also JIT-builds kernels)
#   bash tensorfold-four-spark-tp4/serve.sh smoke       /v1 checks on rank 0 (template ids, stop, thinking split, exact)
#   bash tensorfold-four-spark-tp4/serve.sh bench       the 6 reference prompts, greedy, 512 tokens: ids + tok/s
#   bash tensorfold-four-spark-tp4/serve.sh status | down | logs | tunnel
#   steps of `up` one by one: fadvise | watch | start | wait;   stop = down without stopping the watchdogs
#
# PROFILE=fast-160k   (default) --context 163840, bf16 KV cache whole on every rank: the measured fast path
# PROFILE=dcp4-262k   --context 262144 with decode context parallelism (measured: slower, other bits)
# PROFILE=int4-262k   --context 262144 with an int4 latent cache: PENDING VALIDATION, needs ALLOW_UNVALIDATED=1
# DRY_RUN=1           print every ssh command instead of running it (no ssh at all)
# Any variable from env.sh that you set here (TFS_MIN_AVAIL_GIB, TFS_HTTP_PORT, ...) is forwarded to all four Sparks.
set -uo pipefail

# Captured before env.sh fills in defaults: only what you set yourself is forwarded.
FORWARD_VARS="PROFILE ALLOW_UNVALIDATED RECIPE_HOME VENV TF_REF TF_SRC_DIR B12X_STAGE STATE_DIR MODEL_ROOT PACK_DIR
DRAFTER_DIR HEAD_DIR VIEW_DIR TILES CUDA_HOME FABRIC_IFNAME ROCE_HCA TFS_ROCE TFS_CONTEXT TF_GLM53_DCP
TF_GLM53_CACHE_RESERVE_GB TFS_KV_DTYPE TFS_PROMPT_ROWS TFS_VERIFY_ROWS TFS_DFLASH_DEPTH TFS_DFLASH_CONFIDENCE
TFS_DRAFT_DEFAULT TFS_HEALTH TFS_WARMUP TFS_THINKING TFS_MAX_TOKENS TFS_NAME TFS_ALIAS TFS_HTTP_HOST TFS_HTTP_PORT
TFS_API_KEY_FILE TFS_MASTER_PORT TFS_MIN_START_GIB TFS_MIN_AVAIL_GIB TFS_MAX_SWAP_GROWTH_KB TFS_WATCH_GRACE_S
TF_GLM53_KV_RCP TF_GLM53_KV_TILE TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR BENCH_IDS BENCH_ARGS"
# Sub-steps (up -> fadvise, ...) inherit the top-level list through SERVE_FWD instead of re-reading an environment
# that env.sh has already exported into.
if [[ -n "${SERVE_FWD+x}" ]]; then
  FWD="$SERVE_FWD"
else
  FWD=""
  for k in $FORWARD_VARS; do
    if [[ -n "${!k+x}" ]]; then FWD+=" $k=$(printf %q "${!k}")"; fi
  done
  export SERVE_FWD="$FWD"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
STEP="${1:-}"
DRY="${DRY_RUN:-0}"; [[ "$DRY" == 0 || -z "$DRY" ]] && DRY=0 || DRY=1
read_hosts
REMOTE_REPO="${REMOTE_REPO:-$RECIPE_ROOT}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
[[ -n "${SSH_CONFIG:-}" ]] && SSH+=(-F "$SSH_CONFIG")
TP4_REL="tensorfold-four-spark-tp4"

rcmd() {                                         # rcmd RANK CMD: run CMD in the recipe clone on that rank's Spark
  local r=$1; shift
  local cmd; cmd="cd $(printf %q "$REMOTE_REPO") && env$FWD HOSTS_INLINE=$(printf %q "$HOSTS_INLINE") $*"
  local to=(); [[ -n "${RCMD_TIMEOUT:-}" ]] && to=(timeout "$RCMD_TIMEOUT")
  if [[ "$DRY" == 1 ]]; then echo "DRY [rank $r]: ${H_SSH[$r]}: $cmd"; return 0; fi
  if [[ "${H_SSH[$r]}" == local ]]; then "${to[@]}" bash -c "$cmd"; else "${to[@]}" "${SSH[@]}" "${H_SSH[$r]}" "$cmd"; fi
}
rank_step() { rcmd "$1" "bash $TP4_REL/rank.sh $2 --rank $1"; }

banner() {
  echo "==> profile $PROFILE: context $TFS_CONTEXT, TF_GLM53_DCP $TF_GLM53_DCP, KV $TFS_KV_DTYPE, TensorFold ${TF_REF:0:12}," \
       "DFlash2 d$TFS_DFLASH_DEPTH/c$TFS_DFLASH_CONFIDENCE, RoCE $TFS_ROCE, watchdog floor $TFS_MIN_AVAIL_GIB GiB" \
       "+ swap growth <= $TFS_MAX_SWAP_GROWTH_KB kB" >&2
  if [[ "${PROFILE_VALIDATED:-1}" != 1 ]]; then echo "==> PENDING VALIDATION: nothing about this profile has been measured" >&2; fi
}

wait_ready() {
  local t0 out; t0=$(date +%s)
  while :; do
    out=""
    for r in 0 1 2 3; do out+="rank $r: $(rcmd "$r" "bash $TP4_REL/rank.sh state --rank $r" 2>&1 | tail -n 1)"$'\n'; done
    echo "--- $(date +%T) +$(( $(date +%s) - t0 )) s"; printf '%s' "$out" | cut -c1-220
    case "$out" in
      *FAILED*) echo "START FAILED: stopping all ranks"; bash "$0" stop; return 1 ;;
      *"rank 0: SERVING"*) echo "READY: rank 0 serves /v1 on $TFS_HTTP_HOST:$TFS_HTTP_PORT (model $TFS_NAME)"; return 0 ;;
    esac
    if [[ $(( $(date +%s) - t0 )) -gt "$TFS_START_TIMEOUT_S" ]]; then echo "TIMEOUT"; bash "$0" stop; return 1; fi
    sleep 30
  done
}

case "$STEP" in
preflight)
  banner; check_profile
  mine="$(git -C "$RECIPE_ROOT" rev-parse HEAD 2>/dev/null)"
  ok=1
  for r in 0 1 2 3; do
    if [[ "$DRY" == 0 ]]; then
      rev="$(rcmd "$r" "git rev-parse HEAD" 2>/dev/null | tail -n 1)"
      [[ "$rev" == "$mine" ]] || { echo "rank $r (${H_SSH[$r]}): recipe clone at ${rev:-?}, here ${mine:-?}: git pull on every Spark"; ok=0; }
    fi
    rank_step "$r" preflight || ok=0
  done
  [[ "$DRY" == 1 ]] && exit 0
  echo "preflight $([[ $ok == 1 ]] && echo 'OK on all four' || echo FAILED)"; [[ $ok == 1 ]] ;;
fadvise)
  for r in 0 1 2 3; do rank_step "$r" fadvise; done ;;
watch)
  for r in 1 2 3 0; do rank_step "$r" watch || exit 1; done ;;       # rank 0 last: its central poll sees the peers up
start)
  banner; check_profile
  for r in 1 2 3 0; do                                                # ranks 1-3 first, then rank 0 (PR #159 order)
    if [[ "$DRY" == 1 ]]; then rank_step "$r" start; continue; fi
    RCMD_TIMEOUT=120 rank_step "$r" start \
      || { echo "rank $r on ${H_SSH[$r]} refused / failed: stopping any started ranks"; bash "$0" stop; exit 1; }
  done ;;
wait)
  [[ "$DRY" == 1 ]] && { echo "DRY: poll 'rank.sh state' on all four every 30 s until rank 0 is SERVING (any FAILED stops all)"; exit 0; }
  wait_ready ;;
up)
  banner; check_profile
  bash "$0" fadvise && bash "$0" watch && bash "$0" start && bash "$0" wait ;;
stop)
  for r in 0 1 2 3; do rank_step "$r" stop; done ;;                 # rank 0 first: no new request reaches the others
unwatch)
  for r in 0 1 2 3; do rank_step "$r" unwatch; done ;;
down)
  bash "$0" stop && bash "$0" unwatch ;;
status)
  banner
  for r in 0 1 2 3; do rank_step "$r" status; done ;;
smoke|bench)
  rank_step 0 "$STEP" ;;
logs)
  d="$HERE/runs/$(date +%Y%m%d_%H%M%S)"
  for r in 0 1 2 3; do
    if [[ "$DRY" == 1 ]]; then rank_step "$r" logs-tar; continue; fi
    mkdir -p "$d/rank$r"
    rank_step "$r" logs-tar | tar -xzf - -C "$d/rank$r" 2>/dev/null || echo "rank $r: no logs"
  done
  [[ "$DRY" == 1 ]] || { echo "logs in $d"; find "$d" -type f | sed "s|$d/||"; } ;;
tunnel)
  [[ "${H_SSH[0]}" == local ]] && { echo "rank 0 is this machine: http://127.0.0.1:$TFS_HTTP_PORT/v1"; exit 0; }
  echo "${SSH[*]} -N -L $TFS_HTTP_PORT:127.0.0.1:$TFS_HTTP_PORT ${H_SSH[0]}    # then http://127.0.0.1:$TFS_HTTP_PORT/v1 here" ;;
*)
  sed -n 2,20p "$0"; exit 2 ;;
esac
