#!/usr/bin/env bash
# Host side of the four-Spark server: one Spark, one rank. serve.sh runs it over ssh on each Spark.
#
#   rank.sh preflight  --rank R [--dry-run]   GPU idle, no rank running, MemAvailable, ports, RoCE GID, runtime pin,
#                                             view shas, launcher dry run (rank 0: ssh to the peers for its watchdog)
#   rank.sh fadvise    --rank R [--dry-run]   drop the model files' page cache (drop-model-cache.sh)
#   rank.sh watch      --rank R [--dry-run]   start this Spark's memory / liveness watchdog (rank 0: --central)
#   rank.sh start      --rank R [--dry-run]   start this Spark's rank (refuses without a clean watchdog, on a busy GPU)
#   rank.sh state      --rank R               one line: SERVING | FAILED <why> | LOADING <last log line> | DOWN
#   rank.sh stop       --rank R [--dry-run]   SIGTERM the rank, SIGKILL after 20 s
#   rank.sh unwatch    --rank R [--dry-run]   stop the watchdog
#   rank.sh status     --rank R               rank process, watchdog line, last log lines, /health on rank 0
#   rank.sh smoke|bench --rank 0              bench/bench_v1.py against rank 0's /v1 (results under $STATE_DIR/logs)
#   rank.sh logs-tar   --rank R               tar of this Spark's rank log, watchdog log and bench JSON to stdout
#   rank.sh paths      --rank R               PACK_DIR, DRAFTER_DIR, HEAD_DIR as this Spark resolves them (./glm53 setup)
# --dry-run prints every command instead of running it (start --dry-run also runs the launcher's own --dry-run, which
# reads files only: no GPU, nothing started or killed).
set -uo pipefail
STEP="${1:-}"; shift || true
RANK=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank) RANK="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "rank.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
[[ "$RANK" =~ ^[0-3]$ ]] || { sed -n 2,17p "$0"; exit 2; }
read_hosts
MYIP="${H_IP[$RANK]}"
LOG="$TFS_LOGS/rank$RANK.log"
PAT='[t]f_serve_rank.py'
WPAT='^python3 .*[t]f_serve_memwatch'           # anchored: an ssh command line naming the script must not match
run() { if [[ "$DRY" == 1 ]]; then echo "DRY: $*"; else eval "$@"; fi; }
note() { echo "[$(hostname) rank $RANK] $*"; }

roce_env() {                                    # tools/b12x_env.sh with this rank's fabric address, then the refusals
  [[ "$TFS_ROCE" == 1 ]] || return 0
  # shellcheck source=tools/b12x_env.sh
  FABRIC_IP="$MYIP" B12X_ENV_QUIET=1 source "$HERE/tools/b12x_env.sh" || { echo "REFUSE: tools/b12x_env.sh failed"; return 1; }
  [[ "${NCCL_IB_HCA:-}" == "$ROCE_HCA" ]] || { echo "REFUSE: NCCL_IB_HCA=${NCCL_IB_HCA:-} (needs $ROCE_HCA)"; return 1; }
  [[ -n "${B12X_ROCE_GID_INDEX:-}" ]] || { echo "REFUSE: no RoCE v2 GID for $MYIP on $ROCE_HCA (hosts file fabric_ip?)"; return 1; }
  [[ "${B12X_ROCE_SPIN_LIMIT:-}" == 300000000 ]] || { echo "REFUSE: B12X_ROCE_SPIN_LIMIT=${B12X_ROCE_SPIN_LIMIT:-}"; return 1; }
  return 0
}

launch_env() {                                  # everything the launcher reads, exported
  set -a
  # shellcheck source=env.sh
  source "$HERE/env.sh"
  set +a
  export TFS_MASTER TFS_PEERS
  export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
         TENSORFOLD_NO_UPDATE_CHECK=1
}

port_free() { ! ss -ltnH "( sport = :$1 )" 2>/dev/null | grep -q .; }

case "$STEP" in
preflight)
  ok=1
  ( check_profile ) || ok=0
  if pgrep -f "$PAT" >/dev/null; then note "BUSY: a server rank already runs: $(pgrep -af "$PAT" | cut -c1-120)"; ok=0; fi
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)"
  [[ -z "$apps" ]] || { note "GPU BUSY: $apps"; ok=0; }
  avail="$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)"
  swap="$(awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {print int((t-f)/1024)}' /proc/meminfo)"
  [[ "$avail" -ge "$TFS_MIN_START_GIB" ]] || { note "LOW MEMORY: MemAvailable $avail GiB < $TFS_MIN_START_GIB (./glm53 fadvise, or bash drop-model-cache.sh here; stop other jobs)"; ok=0; }
  if [[ "$RANK" == 0 ]]; then
    for p in "$TFS_MASTER_PORT" $((TFS_MASTER_PORT + 11)) "$TFS_HTTP_PORT"; do
      port_free "$p" || { note "PORT $p in use: $(ss -ltnpH "( sport = :$p )" 2>/dev/null | head -1)"; ok=0; }
    done
    for peer in $TFS_PEERS; do         # the central watchdog must be able to stop the peers
      timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 "$peer" true 2>/dev/null \
        || { note "rank 0 cannot 'ssh -o BatchMode=yes $peer' (the watchdog stops all four through it): ./glm53 init --force on rank 0 prints the fix (or set the hosts file's peer_ssh column)"; ok=0; }
    done
  fi
  ( roce_env && note "RoCE env: HCA=$NCCL_IB_HCA GID=$B12X_ROCE_GID_INDEX SPIN=$B12X_ROCE_SPIN_LIMIT SOCKET=$NCCL_SOCKET_IFNAME fabric $MYIP" ) || ok=0
  ( verify_runtime ) || ok=0
  ( verify_view ) || ok=0
  dry="$( ( launch_env; roce_env >/dev/null 2>&1; "$VENV/bin/python" "$HERE/lib/tf_serve_rank.py" --rank "$RANK" --dry-run ) 2>&1)"
  echo "$dry" | grep -E 'argv:|WOULD REFUSE' | cut -c1-400
  note "launcher dry run: $(echo "$dry" | tail -n 1)"
  case "$dry" in *"dry run ok"*) ;; *) ok=0 ;; esac
  cp_="$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null)"
  note "profile $PROFILE (context $TFS_CONTEXT, DCP $TF_GLM53_DCP, KV $TFS_KV_DTYPE), MemAvailable $avail GiB, swap used $swap MiB, compaction_proactiveness=$cp_"
  note "preflight $([[ $ok == 1 ]] && echo OK || echo FAILED)"
  [[ $ok == 1 ]] ;;
fadvise)
  run "bash '$HERE/drop-model-cache.sh'" ;;
watch)
  if pgrep -f "$WPAT" >/dev/null; then note "watchdog already running"; exit 0; fi
  a=""; [[ "$RANK" == 0 ]] && a="--central"
  run "rm -f /tmp/tf_serve_memwatch.VIOLATION /tmp/tf_serve_memwatch.STOP"
  run "cd /tmp; TFS_PEERS='$TFS_PEERS' TFS_MIN_AVAIL_GIB=$TFS_MIN_AVAIL_GIB TFS_MAX_SWAP_GROWTH_KB=$TFS_MAX_SWAP_GROWTH_KB TFS_WATCH_GRACE_S=$TFS_WATCH_GRACE_S nohup setsid python3 '$HERE/lib/tf_serve_memwatch.py' $a > /tmp/tf_serve_memwatch.out 2>&1 < /dev/null &"
  [[ "$DRY" == 1 ]] && exit 0
  sleep 2
  pgrep -f "$WPAT" >/dev/null || { note "REFUSE: watchdog did not start: $(tail -3 /tmp/tf_serve_memwatch.out)"; exit 1; }
  note "watchdog: $(grep '"event": "start"' /tmp/tf_serve_memwatch.log | tail -1 | cut -c1-200)" ;;
start)
  check_profile
  if [[ "$DRY" == 0 ]]; then
    pgrep -f "$PAT" >/dev/null && { note "REFUSE: a server rank already runs"; exit 1; }
    pgrep -f "$WPAT" >/dev/null || { note "REFUSE: no watchdog (rank.sh watch first; serve.sh up does it)"; exit 1; }
    [[ -e /tmp/tf_serve_memwatch.VIOLATION ]] && { note "REFUSE: watchdog flagged $(cat /tmp/tf_serve_memwatch.VIOLATION)"; exit 1; }
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)"
    [[ -z "$apps" ]] || { note "REFUSE: GPU busy: $apps"; exit 1; }
    avail="$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)"
    [[ "$avail" -ge "$TFS_MIN_START_GIB" ]] || { note "REFUSE: MemAvailable $avail GiB < $TFS_MIN_START_GIB"; exit 1; }
  fi
  launch_env
  roce_env || [[ "$DRY" == 1 ]] || exit 1
  CMD="'$VENV/bin/python' '$HERE/lib/tf_serve_rank.py' --rank $RANK"
  if [[ "$DRY" == 1 ]]; then
    echo "DRY: cd '$HERE/lib'; nohup setsid $CMD > '$LOG' 2>&1 < /dev/null &"
    "$VENV/bin/python" "$HERE/lib/tf_serve_rank.py" --rank "$RANK" --dry-run
    exit $?
  fi
  mkdir -p "$TFS_LOGS" "$TFS_CTL"
  [[ -e "$LOG" ]] && mv "$LOG" "$LOG.prev.$(date +%s)"
  cd "$HERE/lib" || exit 1
  eval "nohup setsid $CMD > '$LOG' 2>&1 < /dev/null &"
  sleep 1
  note "started: $(pgrep -af "$PAT" | cut -c1-140) log $LOG gid=${B12X_ROCE_GID_INDEX:-n/a}" ;;
state)
  v="$(cat /tmp/tf_serve_memwatch.VIOLATION 2>/dev/null)"
  bad="$(grep -a -h -E 'FATAL|REFUSE|Traceback|EXIT [0-9]|rank-order sum: False' "$LOG" 2>/dev/null | head -2 | cut -c1-200)"
  if [[ -n "$v" ]]; then echo "FAILED watchdog: $v"
  elif [[ -n "$bad" ]]; then echo "FAILED $bad"
  elif [[ "$RANK" == 0 ]] && grep -a -q -E '\[tensorfold\] serving .*/v1' "$LOG" 2>/dev/null; then echo "SERVING $(grep -a -m1 -E '\[tensorfold\] serving .*/v1' "$LOG" | cut -c1-200)"
  elif pgrep -f "$PAT" >/dev/null; then echo "LOADING $(tail -n 1 "$LOG" 2>/dev/null | cut -c1-160)"
  else echo "DOWN"; fi ;;
stop)
  pids="$(pgrep -f "$PAT")"
  [[ -z "$pids" ]] && { note "no server rank running"; exit 0; }
  run "kill -TERM $pids"
  [[ "$DRY" == 1 ]] && exit 0
  for _ in $(seq 1 20); do pgrep -f "$PAT" >/dev/null || break; sleep 1; done
  left="$(pgrep -f "$PAT")"
  [[ -n "$left" ]] && { note "SIGKILL $left after 20 s"; kill -KILL $left; sleep 1; }
  note "stopped; left: $(pgrep -fc "$PAT")" ;;
unwatch)
  run "touch /tmp/tf_serve_memwatch.STOP"
  [[ "$DRY" == 1 ]] && exit 0
  sleep 2; note "watchdog $(pgrep -f "$WPAT" >/dev/null && echo 'still running' || echo stopped): $(tail -n 1 /tmp/tf_serve_memwatch.log 2>/dev/null | cut -c1-200)" ;;
status)
  note "rank procs: $(pgrep -fc "$PAT")  watchdog: $(pgrep -f "$WPAT" >/dev/null && echo up || echo down)  $(cat /tmp/tf_serve_memwatch.VIOLATION 2>/dev/null)"
  tail -n 1 /tmp/tf_serve_memwatch.log 2>/dev/null | cut -c1-220
  grep -a -E '\[tf_serve\]|serving |Traceback|Error|rank-order sum|RoCE|READY|FATAL|EXIT' "$LOG" 2>/dev/null | tail -n 6 | cut -c1-240
  [[ "$RANK" == 0 ]] && echo "health: $(curl -s -m 5 "http://127.0.0.1:$TFS_HTTP_PORT/health" 2>&1 | cut -c1-300)" ;;
smoke|bench)
  [[ "$RANK" == 0 ]] || { echo "smoke/bench run on rank 0"; exit 2; }
  mkdir -p "$TFS_LOGS"
  out="$TFS_LOGS/${STEP}_$(date +%Y%m%d_%H%M%S).json"
  args=(--base "http://127.0.0.1:$TFS_HTTP_PORT/v1" --model "$TFS_NAME" --ref "$RECIPE_ROOT/bench/sweep_reference.json" --out "$out")
  [[ -n "$TFS_API_KEY_FILE" ]] && args+=(--api-key "$(cat "$TFS_API_KEY_FILE")")
  if [[ "$STEP" == smoke ]]; then args+=(--smoke); else args+=(--ids "$BENCH_IDS"); fi
  # shellcheck disable=SC2206
  args+=(${BENCH_ARGS:-})
  if [[ "$DRY" == 1 ]]; then echo "DRY: python3 $RECIPE_ROOT/bench/bench_v1.py ${args[*]}"; exit 0; fi
  python3 "$RECIPE_ROOT/bench/bench_v1.py" "${args[@]}" ;;
paths)
  printf '%s\n' "$PACK_DIR" "$DRAFTER_DIR" "$HEAD_DIR" ;;
logs-tar)
  files=()
  for f in "$LOG" /tmp/tf_serve_memwatch.log /tmp/tf_serve_memwatch.VIOLATION "$TFS_LOGS"/smoke_*.json "$TFS_LOGS"/bench_*.json; do
    [[ -e "$f" ]] && files+=("$f")
  done
  [[ ${#files[@]} -gt 0 ]] && tar -czf - "${files[@]}" 2>/dev/null ;;
*)
  sed -n 2,17p "$0"; exit 2 ;;
esac
